// ============================================================
// CMS Stack Plot Macro — Data/MC + Ratio
// e + jets channel, Run-2 UL
//
// Usage:
//   root -l -b -q 'plot_stacked_histograms.C("2018")'
//   root -l -b -q 'plot_stacked_histograms.C("2017")'
//   root -l -b -q 'plot_stacked_histograms.C("2016preVFP")'
//   root -l -b -q 'plot_stacked_histograms.C("2016postVFP")'
//
// ROOT file layout expected (written by coffea->ROOT converter):
//   hist_name/dataset   e.g.  electron_pt/ttbar_SemiLeptonic
//                             electron_pt/DATA
// ============================================================

#include <TFile.h>
#include <TDirectory.h>
#include <TH1D.h>
#include <THStack.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TSystem.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ============================================================
// Data structures
// ============================================================

struct EraInfo {
    double lumi;        // integrated luminosity in fb^-1
    string lumiLabel;   // string for the plot header
};

struct ProcessGroup {
    string         name;      // internal key
    vector<string> datasets;  // dataset keys in ROOT file
    int            color;     // ROOT fill color
    string         label;     // legend label
    TH1D*          hist;      // summed normalized histogram

    ProcessGroup(const string& n, const vector<string>& ds, int c, const string& l)
        : name(n), datasets(ds), color(c), label(l), hist(nullptr) {}
};

// ============================================================
// x-axis labels
// ============================================================
map<string,string> AxisLabels() {
    map<string,string> m;
    m["electron_pt"]  = "p_{T}^{e} [GeV]";
    m["electron_eta"] = "#eta^{e}";
    m["electron_phi"] = "#phi^{e} [rad]";
    m["Jet_pt"]       = "p_{T}^{jet} [GeV]";
    m["Jet_eta"]      = "#eta^{jet}";
    m["Jet_phi"]      = "#phi^{jet} [rad]";
    m["met_pt"]       = "p_{T}^{miss} [GeV]";
    m["met_phi"]      = "#phi^{miss} [rad]";
    m["nJets"]        = "Jet multiplicity";
    m["nBJets"]       = "b-jet multiplicity";
    m["HT"]           = "H_{T} [GeV]";
    m["mttbar"]       = "m_{t#bar{t}} [GeV]";
    m["top_pt"]       = "p_{T}^{top} [GeV]";
    m["top_mass"]     = "m_{top} [GeV]";
    m["pTSum"]        = "#Sigma p_{T} [GeV]";
    m["AL"]           = "Jet asymm.";
    m["planarity"]    = "alignment";
    m["Sxz"]          = "S_{xz}" ;
    m["Szz"]          = "S_{zz}";
    m["p2in"]         = "p2in";
    m["nJet"]         = "Jet_multiplicity";
    m["delta_R"]      = "#DeltaR(e, jet_system)";
    m["FW1"]          = "Fox-Wolfram H_{1}";
    return m;
}

// ============================================================
// CMS style
// ============================================================
void SetCMSStyle() {
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetFrameFillColor(0);
    gStyle->SetFrameFillStyle(0);
    gStyle->SetFrameLineColor(kBlack);
    gStyle->SetFrameLineWidth(1);
    gStyle->SetPadGridX(false);
    gStyle->SetPadGridY(false);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetTickLength(0.03, "XYZ");
    gStyle->SetNdivisions(510, "XYZ");
    gStyle->SetLabelFont(42, "XYZ");
    gStyle->SetLabelSize(0.07, "XYZ");
    gStyle->SetLabelColor(kBlack, "XYZ");
    gStyle->SetLabelOffset(0.007, "XYZ");
    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetTitleSize(0.067, "XYZ");
    gStyle->SetTitleColor(kBlack, "XYZ");
    gStyle->SetTitleXOffset(0.9);
    gStyle->SetTitleYOffset(1.25);
    gStyle->SetMarkerStyle(20);
    gStyle->SetMarkerSize(1.0);
    gStyle->SetEndErrorSize(2);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(kWhite);
    gStyle->SetLegendFont(42);
    gROOT->ForceStyle();
}

// ============================================================
// CMS header — era-aware
// ============================================================
void DrawCMSHeader(TPad* pad, const EraInfo& era) {
    pad->cd();
    TLatex lat;
    lat.SetNDC();
    lat.SetTextColor(kBlack);
    lat.SetTextAngle(0);

    lat.SetTextFont(61);
    lat.SetTextSize(0.07);
    lat.SetTextAlign(11);
    lat.DrawLatex(0.18, 0.84, "CMS");

    lat.SetTextFont(52);
    lat.SetTextSize(0.06);
    lat.SetTextAlign(11);
    lat.DrawLatex(0.30, 0.84, "Preliminary");

    lat.SetTextFont(52);
    lat.SetTextSize(0.05);
    lat.SetTextAlign(11);
    lat.DrawLatex(0.25, 0.79, "e + jets");

    lat.SetTextFont(42);
    lat.SetTextSize(0.05);
    lat.SetTextAlign(31);
    lat.DrawLatex(0.95, 0.91, era.lumiLabel.c_str());
}

// ============================================================
// Load a TH1D from a ROOT file using explicit two-step directory
// navigation. This is necessary because uproot writes a same-named
// top-level TH1D alongside each subdirectory, making single-step
// path resolution ("histname/dataset") ambiguous in ROOT.
// Returns a detached clone or nullptr on failure.
// ============================================================
TH1D* LoadHist(const string& filename,
               const string& histname,
               const string& dataset) {
    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) {
        cerr << "ERROR: cannot open " << filename << endl;
        return nullptr;
    }
    TDirectory* dir = dynamic_cast<TDirectory*>(f->Get(histname.c_str()));
    if (!dir) {
        f->Close(); delete f;
        return nullptr;
    }
    TH1D* h = dynamic_cast<TH1D*>(dir->Get(dataset.c_str()));
    if (!h) {
        f->Close(); delete f;
        return nullptr;
    }
    TH1D* clone = dynamic_cast<TH1D*>(
        h->Clone(("clone_" + histname + "_" + dataset).c_str()));
    clone->SetDirectory(0);
    f->Close(); delete f;
    return clone;
}

// ============================================================
// Apply bin-by-bin QCD transfer factors
// ============================================================
void ApplyTransferFactors(TH1D* h, const vector<double>& tf) {
    if (!h || tf.empty()) return;
    int nb = h->GetNbinsX();
    for (int i = 1; i <= nb; ++i) {
        double sf = (i <= (int)tf.size()) ? tf[i-1] : 1.0;
        h->SetBinContent(i, h->GetBinContent(i) * sf);
        h->SetBinError  (i, h->GetBinError(i)   * sf);
    }
}

// ============================================================
// Load JSON transfer factors
// ============================================================
map<string, vector<double>> LoadTransferFactors(const string& filename) {
    map<string, vector<double>> result;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "WARNING: cannot open " << filename << " — QCD unscaled." << endl;
        return result;
    }
    ostringstream buf;
    buf << file.rdbuf();
    string s = buf.str();
    s.erase(remove_if(s.begin(), s.end(), ::isspace), s.end());

    size_t pos = 0;
    while (pos < s.size()) {
        size_t qs = s.find('"', pos);       if (qs == string::npos) break;
        size_t qe = s.find('"', qs+1);      if (qe == string::npos) break;
        string key = s.substr(qs+1, qe-qs-1);
        size_t col = s.find(':', qe);        if (col == string::npos) break;
        size_t as  = s.find('[', col);       if (as  == string::npos) break;
        size_t ae  = s.find(']', as);        if (ae  == string::npos) break;
        string arr = s.substr(as+1, ae-as-1);
        vector<double> vals;
        stringstream ss(arr); string tok;
        while (getline(ss, tok, ','))
            if (!tok.empty()) vals.push_back(stod(tok));
        result[key] = vals;
        pos = ae + 1;
    }
    cout << "Transfer factors loaded: " << result.size() << " histograms." << endl;
    return result;
}

// ============================================================
// Main plotting function — one call per histogram name
// ============================================================
void PlotHistogram(
    const string&                     histname,
    const string&                     mc_file,
    const string&                     qcd_file,
    const string&                     data_file,
    const map<string,double>&         norm_factors,
    const map<string,vector<double>>& tf_map,
    const map<string,string>&         axis_labels,
    vector<ProcessGroup>&             groups,
    const EraInfo&                    era,
    const string&                     era_tag,
    TFile*                            outfile)
{
    cout << "\n" << string(60,'=') << "\n"
         << "  " << histname << "\n"
         << string(60,'=') << endl;

    // ----------------------------------------------------------
    // Reset all group histograms from the previous histogram call
    // ----------------------------------------------------------
    for (auto& g : groups) {
        delete g.hist;
        g.hist = nullptr;
    }

    // ----------------------------------------------------------
    // Build summed normalized histogram per group
    // ----------------------------------------------------------
    for (auto& g : groups) {
        for (const string& ds : g.datasets) {
            TH1D* h = nullptr;

            if (ds == "QCD") {
                h = LoadHist(qcd_file, histname, "QCD");
                if (!h) { cout << "  MISS: QCD/" << histname << endl; continue; }
                auto it = tf_map.find(histname);
                if (it != tf_map.end())
                    ApplyTransferFactors(h, it->second);
                else
                    cout << "  WARNING: no transfer factors for " << histname << endl;
            } else {
                h = LoadHist(mc_file, histname, ds);
                if (!h) { cout << "  MISS: " << ds << "/" << histname << endl; continue; }
                auto nit = norm_factors.find(ds);
                if (nit != norm_factors.end())
                    h->Scale(nit->second);
                else
                    cout << "  WARNING: no norm factor for " << ds << endl;
            }

            if (!g.hist) {
                g.hist = dynamic_cast<TH1D*>(
                    h->Clone(("grp_" + g.name).c_str()));
                g.hist->SetDirectory(0);
            } else {
                g.hist->Add(h);
            }
            delete h;
        }

        if (g.hist)
            cout << "  " << left << setw(22) << g.label
                 << "  integral = " << fixed << setprecision(1)
                 << g.hist->Integral() << endl;
    }

    // ----------------------------------------------------------
    // Data histogram
    // ----------------------------------------------------------
    TH1D* data = LoadHist(data_file, histname, "DATA");
    if (!data)
        cout << "  WARNING: DATA not found for " << histname << endl;
    else
        cout << "  " << left << setw(22) << "Data"
             << "  integral = " << fixed << setprecision(1)
             << data->Integral() << endl;

    // ----------------------------------------------------------
    // Sort groups by integral ascending — smallest at bottom of stack.
    // Index permutation preserves groups vector order across calls.
    // ----------------------------------------------------------
    vector<int> order((int)groups.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        double ia = groups[a].hist ? groups[a].hist->Integral() : 0.0;
        double ib = groups[b].hist ? groups[b].hist->Integral() : 0.0;
        return ia < ib;
    });

    // ----------------------------------------------------------
    // Total MC = sum of all groups
    // ----------------------------------------------------------
    TH1D* total_mc = nullptr;
    for (int idx : order) {
        if (!groups[idx].hist) continue;
        if (!total_mc) {
            total_mc = dynamic_cast<TH1D*>(
                groups[idx].hist->Clone(("totalmc_" + histname).c_str()));
            total_mc->SetDirectory(0);
        } else {
            total_mc->Add(groups[idx].hist);
        }
    }

    if (!total_mc) {
        cout << "  SKIP: no MC histograms available for " << histname << endl;
        delete data;
        return;
    }

    // ----------------------------------------------------------
    // y-axis range — generous headroom for single-column legend
    // ----------------------------------------------------------
    double ymax = total_mc->GetMaximum();
    if (data && data->GetMaximum() > ymax) ymax = data->GetMaximum();
    const double ymin     = 0.1;
    const double yplotmax = ymax * 9000.0;  // large headroom for single-col legend

    // ----------------------------------------------------------
    // x-axis label
    // ----------------------------------------------------------
    auto xit = axis_labels.find(histname);
    const string xlabel = (xit != axis_labels.end()) ? xit->second : histname;

    // ----------------------------------------------------------
    // Canvas: 800x900, split 70% upper / 30% lower
    // ----------------------------------------------------------
    TCanvas* c = new TCanvas(("c_" + histname).c_str(), histname.c_str(), 800, 900);
    c->SetFillColor(kWhite);

    TPad* pad1 = new TPad("pad1", "", 0.0, 0.30, 1.0, 1.0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.10);
    pad1->SetBottomMargin(0.02);
    pad1->SetLeftMargin(0.16);
    pad1->SetRightMargin(0.05);
    pad1->SetLogy();
    pad1->SetTicks(1, 1);
    pad1->Draw();

    c->cd();
    TPad* pad2 = new TPad("pad2", "", 0.0, 0.0, 1.0, 0.30);
    pad2->SetFillColor(kWhite);
    pad2->SetTopMargin(0.03);
    pad2->SetBottomMargin(0.35);
    pad2->SetLeftMargin(0.16);
    pad2->SetRightMargin(0.05);
    pad2->SetTicks(1, 1);
    pad2->Draw();

    // ==========================================================
    // PAD 1 — stacked MC + data overlay
    // ==========================================================
    pad1->cd();

    THStack*      stack       = new THStack(("stack_" + histname).c_str(), "");
    vector<TH1D*> stack_owned;

    for (int idx : order) {
        if (!groups[idx].hist) continue;
        TH1D* hcopy = dynamic_cast<TH1D*>(
            groups[idx].hist->Clone(("sc_" + groups[idx].name).c_str()));
        hcopy->SetDirectory(0);
        hcopy->SetFillColor(groups[idx].color);
        hcopy->SetFillStyle(1001);
        hcopy->SetLineColor(kBlack);
        hcopy->SetLineWidth(1);
        stack->Add(hcopy);
        stack_owned.push_back(hcopy);
    }

    // Draw("HIST") — never "HIST SAME" — THStack must create its own axes
    stack->Draw("HIST");
    stack->SetMinimum(ymin);
    stack->SetMaximum(yplotmax);
    stack->GetXaxis()->SetLabelSize(0);
    stack->GetXaxis()->SetTitleSize(0);

    // y-axis: tested values
    stack->GetYaxis()->SetTitle("Events");
    stack->GetYaxis()->SetTitleFont(42);
    stack->GetYaxis()->SetTitleSize(0.07);
    stack->GetYaxis()->SetTitleOffset(1.18);
    stack->GetYaxis()->SetLabelFont(42);
    stack->GetYaxis()->SetLabelSize(0.07);
    stack->GetYaxis()->SetLabelOffset(0.004);
    stack->GetYaxis()->SetTickLength(0.04);
    stack->GetYaxis()->SetMoreLogLabels(kFALSE);
    stack->GetYaxis()->SetNoExponent(kFALSE);

    // MC statistical uncertainty band
    TH1D* mc_unc = dynamic_cast<TH1D*>(
        total_mc->Clone(("mcunc_" + histname).c_str()));
    mc_unc->SetDirectory(0);
    mc_unc->SetFillStyle(3354); //3244 before
    mc_unc->SetFillColor(kGray+2);
    mc_unc->SetLineColor(kGray+2);
    mc_unc->SetLineWidth(0);
    mc_unc->SetMarkerSize(0);
    mc_unc->Draw("E2 SAME");

    // Redraw frame axes so tick marks are not buried under fills
    stack->GetHistogram()->Draw("AXIS SAME");

    // Data points
    if (data) {
        data->SetMarkerStyle(20);
        data->SetMarkerSize(1.2);   // tested
        data->SetMarkerColor(kBlack);
        data->SetLineColor(kBlack);
        data->SetLineWidth(2);      // tested
        data->Draw("E1 SAME");
    }

    // Legend — single column
    TLegend* leg = new TLegend(0.55, 0.55, 0.92, 0.88);
    leg->SetNColumns(1);
    leg->SetTextFont(42);
    leg->SetTextSize(0.029);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->SetBorderSize(0);

    if (data) leg->AddEntry(data, "Data", "lep");

    // Stat uncertainty entry
    TH1D* unc_leg = new TH1D(("uncleg_" + histname).c_str(), "", 1, 0, 1);
    unc_leg->SetDirectory(0);
    unc_leg->SetFillStyle(3354);
    unc_leg->SetFillColor(kGray+3);
    unc_leg->SetLineColor(kGray+3);
    unc_leg->SetLineWidth(0);
    unc_leg->SetMarkerSize(0);
    leg->AddEntry(unc_leg, "Stat. unc.", "f");

    // Process entries — largest first (reverse of stack order)
    vector<TH1D*> leg_dummies;
    for (int i = (int)order.size()-1; i >= 0; --i) {
        ProcessGroup& g = groups[order[i]];
        if (!g.hist) continue;
        TH1D* d = new TH1D(("legdummy_" + g.name + "_" + histname).c_str(), "", 1, 0, 1);
        d->SetDirectory(0);
        d->SetFillColor(g.color);
        d->SetFillStyle(1001);
        d->SetLineColor(kBlack);
        d->SetLineWidth(1);
        leg->AddEntry(d, g.label.c_str(), "f");
        leg_dummies.push_back(d);
    }

    leg->Draw();
    DrawCMSHeader(pad1, era);

    pad1->Modified();
    pad1->Update();

    // ==========================================================
    // PAD 2 — Data / Pred. ratio
    // ==========================================================
    pad2->cd();

    TH1D* ratio     = nullptr;
    TH1D* ratio_unc = nullptr;

    if (data) {
        ratio = dynamic_cast<TH1D*>(data->Clone(("ratio_" + histname).c_str()));
        ratio->SetDirectory(0);
        ratio->Divide(total_mc);

        ratio->SetMarkerStyle(20);
        ratio->SetMarkerSize(1.2);    // tested
        ratio->SetMarkerColor(kBlack);
        ratio->SetLineColor(kBlack);
        ratio->SetLineWidth(2);       // tested

        // y-axis: tested values
        ratio->GetYaxis()->SetTitle("Data / MC");
        ratio->GetYaxis()->SetRangeUser(0.5, 1.5);
        ratio->GetYaxis()->SetNdivisions(505);
        ratio->GetYaxis()->SetTitleFont(42);
        ratio->GetYaxis()->SetTitleSize(0.15);
        ratio->GetYaxis()->SetTitleOffset(0.55);
        ratio->GetYaxis()->SetLabelFont(42);
        ratio->GetYaxis()->SetLabelSize(0.16);
        ratio->GetYaxis()->SetLabelOffset(0.006);
        ratio->GetYaxis()->SetTickLength(0.06);

        // x-axis: tested values
        ratio->GetXaxis()->SetTitle(xlabel.c_str());
        ratio->GetXaxis()->SetTitleFont(42);
        ratio->GetXaxis()->SetTitleSize(0.18);
        ratio->GetXaxis()->SetTitleOffset(0.87);
        ratio->GetXaxis()->SetLabelFont(42);
        ratio->GetXaxis()->SetLabelSize(0.16);
        ratio->GetXaxis()->SetLabelOffset(0.006);
        ratio->GetXaxis()->SetTickLength(0.12);

        // MC stat uncertainty band centred at 1
        ratio_unc = dynamic_cast<TH1D*>(
            total_mc->Clone(("ratiounc_" + histname).c_str()));
        ratio_unc->SetDirectory(0);
        for (int i = 1; i <= ratio_unc->GetNbinsX(); ++i) {
            double mc_val = total_mc->GetBinContent(i);
            double mc_err = total_mc->GetBinError(i);
            if (mc_val > 0.0) {
                ratio_unc->SetBinContent(i, 1.0);
                ratio_unc->SetBinError  (i, mc_err / mc_val);
            } else {
                ratio_unc->SetBinContent(i, 0.0);
                ratio_unc->SetBinError  (i, 0.0);
            }
        }
        ratio_unc->SetFillStyle(3354);
        ratio_unc->SetFillColor(kGray+7);
        ratio_unc->SetLineColor(kGray+7);
        ratio_unc->SetLineWidth(0);
        ratio_unc->SetMarkerSize(0);

        // Draw order: ratio sets axes, band on top, ratio redrawn last
        ratio->Draw("E1");
        ratio_unc->Draw("E2 SAME");
        ratio->Draw("E1 SAME");

        // Unity line
        double xlo = ratio->GetXaxis()->GetXmin();
        double xhi = ratio->GetXaxis()->GetXmax();
        TLine* unity = new TLine(xlo, 1.0, xhi, 1.0);
        unity->SetLineColor(kRed+1);
        unity->SetLineStyle(2);
        unity->SetLineWidth(4);   // tested
        unity->Draw();

        // ±20% guide lines
        for (double ref : {0.6,0.8, 1.2,1.4}) {
            TLine* gl = new TLine(xlo, ref, xhi, ref);
            gl->SetLineColor(kGray+2);
            gl->SetLineStyle(3);
            gl->SetLineWidth(2);
            gl->Draw();
        }
        // TLine objects owned by pad2 after Draw()
    }

    pad2->Modified();
    pad2->Update();

    // ==========================================================
    // Save — flush all pads before SaveAs.
    // All drawn objects must remain alive until after SaveAs/Write.
    // ==========================================================
    gSystem->mkdir("plots", kTRUE);
    pad1->cd(); pad1->Modified(); pad1->Update();
    pad2->cd(); pad2->Modified(); pad2->Update();
    c->cd();    c->Modified();    c->Update();

    const string outpng = "plots/" + histname + "_" + era_tag + ".png";
    const string outpdf = "plots/" + histname + "_" + era_tag + ".pdf";
    c->SaveAs(outpng.c_str());
    c->SaveAs(outpdf.c_str());
    cout << "  Saved: " << outpng << "  and  " << outpdf << endl;

    if (outfile && !outfile->IsZombie()) {
        outfile->cd();
        c->Write(histname.c_str());
    }

    // ==========================================================
    // Summary printout
    // ==========================================================
    const string SEP1(68, '=');
    const string SEP2(68, '-');
    cout << "\n" << SEP1 << "\n  SUMMARY: " << histname << "\n" << SEP1 << "\n";
    if (data)
        cout << "  Data       : " << fixed << setprecision(1)
             << data->Integral() << "\n";
    cout << "  Total MC   : " << fixed << setprecision(1)
         << total_mc->Integral() << "\n";
    if (data && total_mc->Integral() > 0)
        cout << "  Data/Pred. : " << fixed << setprecision(3)
             << data->Integral() / total_mc->Integral() << "\n";
    cout << SEP2 << "\n";
    cout << "  " << left  << setw(24) << "Process"
         << right << setw(14) << "Events"
         << setw(10) << "% of MC" << "\n" << SEP2 << "\n";
    for (int i = (int)order.size()-1; i >= 0; --i) {
        if (!groups[order[i]].hist) continue;
        double intg = groups[order[i]].hist->Integral();
        double frac = 100.0 * intg / total_mc->Integral();
        cout << "  " << left  << setw(24) << groups[order[i]].label
             << right << setw(14) << fixed << setprecision(1) << intg
             << setw(9)  << setprecision(1) << frac << " %\n";
    }
    cout << SEP1 << "\n";

    // ==========================================================
    // Cleanup — canvas first, then stack, then everything else
    // ==========================================================
    delete c;
    delete stack;
    for (TH1D* h : stack_owned) delete h;
    delete mc_unc;
    delete unc_leg;
    for (TH1D* d : leg_dummies) delete d;
    delete leg;
    delete ratio;
    delete ratio_unc;
    delete total_mc;
    delete data;
}

// ============================================================
// MAIN
// ============================================================
void plot_stacked_histograms_thesis(const char* era_arg = "2018") {

    SetCMSStyle();

    const string ERA(era_arg);

    // Era metadata
    map<string,EraInfo> era_map;
    era_map["2016preVFP"]  = { 19.5212, "19.5 fb^{-1} (13 TeV, 2016 preVFP)"  };
    era_map["2016postVFP"] = { 16.8121, "16.8 fb^{-1} (13 TeV, 2016 postVFP)" };
    era_map["2017"]        = { 41.4796, "41.5 fb^{-1} (13 TeV, 2017)"          };
    era_map["2018"]        = { 59.2227, "59.2 fb^{-1} (13 TeV, 2018)"          };

    if (!era_map.count(ERA)) {
        cerr << "ERROR: unknown era '" << ERA << "'\n"
             << "Valid: 2016preVFP  2016postVFP  2017  2018" << endl;
        return;
    }
    const EraInfo& era = era_map[ERA];
    cout << "Era: " << ERA << "  Lumi: " << era.lumi << " fb^-1" << endl;

    // Generated event counts [era][dataset]
    map<string, map<string,double>> ngen_map;

    ngen_map["2016preVFP"] = {
        {"ttbar_SemiLeptonic",  131106831},
        {"ttbar_FullyLeptonic",  37202073},
        {"Tchannel",             52437432},
        {"Tbarchannel",          29205915},
        {"Schannel",              3592772},
        {"tw_top",                3294485},
        {"tw_antitop",            3176335},
        {"DYJetsToLL",           95170552},
        {"WJetsToLNu_0J",       121208493},
        {"WJetsToLNu_1J",        84198168},
        {"WJetsToLNu_2J",        27463756},
        {"WWTo2L2Nu",             3006596},
        {"WWTolnulnu",            4932000},
        {"WZTo2Q2L",              9780392},
        {"ZZTo2L2Nu",            16826232},
        {"ZZTo2Q2L",             10406942},
    };
    ngen_map["2016postVFP"] = {
        {"ttbar_SemiLeptonic",  144722000},
        {"ttbar_FullyLeptonic",  43546000},
        {"Tchannel",             63073000},
        {"Tbarchannel",          30609000},
        {"Schannel",              5471000},
        {"tw_top",                3368375},
        {"tw_antitop",            3654510},
        {"DYJetsToLL",           82448544},
        {"WJetsToLNu_0J",       159756701},
        {"WJetsToLNu_1J",       167292982},
        {"WJetsToLNu_2J",        26790000},
        {"WWTo2L2Nu",             2900000},
        {"WWTolnulnu",            4932000},
        {"WZTo2Q2L",             13526954},
        {"ZZTo2L2Nu",            16826232},
        {"ZZTo2Q2L",             13740600},
    };
    ngen_map["2017"] = {
        {"ttbar_SemiLeptonic",  343257745},
        {"ttbar_FullyLeptonic", 105860011},
        {"Tchannel",            121728258},
        {"Tbarchannel",          65701149},
        {"Schannel",              8866570},
        {"tw_top",                8506765},
        {"tw_antitop",            8433562},
        {"DYJetsToLL",          131552392},
        {"WJetsToLNu_0J",       135263983},
        {"WJetsToLNu_1J",        85950236},
        {"WJetsToLNu_2J",        29987306},
        {"WWTo2L2Nu",             7071358},
        {"WWTolnulnu",            2000000},
        {"WZTo2Q2L",             18136497},
        {"ZZTo2L2Nu",            16826232},
        {"ZZTo2Q2L",             19134840},
    };
    ngen_map["2018"] = {
        {"ttbar_SemiLeptonic",  472977862},
        {"ttbar_FullyLeptonic", 143830836},
        {"Tchannel",            166637158},
        {"Tbarchannel",          89985007},
        {"Schannel",             12444591},
        {"tw_top",               11270430},
        {"tw_antitop",           10949620},
        {"DYJetsToLL",          129037134},
        {"WJetsToLNu_0J",       137259710},
        {"WJetsToLNu_1J",        87594835},
        {"WJetsToLNu_2J",        29028341},
        {"WWTo2L2Nu",             9962019},
        {"WZTo2Q2L",             17952068},
        {"ZZTo2L2Nu",            16826232},
        {"ZZTo2Q2L",             19082659},
        // WWTolnulnu not present in 2018
    };

    // Cross sections [pb]
    map<string,double> xsec_map = {
        {"ttbar_SemiLeptonic",   366.3     },
        {"ttbar_FullyLeptonic",   88.5     },
        {"Tchannel",             134.2     },
        {"Tbarchannel",           80.0     },
        {"Schannel",               2.215836},
        {"tw_top",                39.65    },
        {"tw_antitop",            39.65    },
        {"DYJetsToLL",          6424.0     },
        {"WJetsToLNu_0J",      52780.0     },
        {"WJetsToLNu_1J",       8832.0     },
        {"WJetsToLNu_2J",       3276.0     },
        {"WWTo2L2Nu",             11.09    },
        {"WWTolnulnu",            10.48    },
        {"WZTo2Q2L",               6.565   },
        {"ZZTo2L2Nu",              0.974   },
        {"ZZTo2Q2L",               3.676   },
    };
    if (ERA == "2017") {
        xsec_map["WWTolnulnu"] = 10.79;
        xsec_map["WZTo2Q2L"]   =  5.595;
    }

    // Normalization factors: xsec [pb] * lumi [fb^-1] * 1000 / ngen
    const map<string,double>& ngen = ngen_map[ERA];
    map<string,double> norm_factors;
    for (const auto& kv : ngen) {
        auto xit = xsec_map.find(kv.first);
        if (xit == xsec_map.end()) {
            cerr << "WARNING: no xsec for " << kv.first << endl;
            continue;
        }
        norm_factors[kv.first] = (xit->second * era.lumi * 1000.0) / kv.second;
    }

    // Input files
    const string mc_file   = "regionD_ABCD_for_" + ERA + "_with_nbtags_vs_id.root";
    const string qcd_file  = "QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.root";
    const string data_file = "regionD_ABCD_for_" + ERA + "_with_nbtags_vs_id.root";

    // Transfer factors for QCD ABCD method
    const map<string,vector<double>> tf_map =
        LoadTransferFactors("bin_by_bin_regionC_by_regionA.json");

    // Process groups
    vector<ProcessGroup> groups;
    groups.emplace_back("QCD",
        vector<string>{"QCD"},
        kYellow-7, "QCD (dd)");
    groups.emplace_back("WJets",
        vector<string>{"WJetsToLNu_0J","WJetsToLNu_1J","WJetsToLNu_2J"},
        kGreen+1, "W+jets");
    groups.emplace_back("DYJets",
        vector<string>{"DYJetsToLL"},
        kCyan+1, "DY+jets");
    {
        vector<string> diboson = {"WWTo2L2Nu","WZTo2Q2L","ZZTo2L2Nu","ZZTo2Q2L"};
        if (ERA != "2018") diboson.push_back("WWTolnulnu");
        groups.emplace_back("Diboson", diboson, kViolet-3, "Diboson");
    }
    groups.emplace_back("SingleTop",
        vector<string>{"Tchannel","Tbarchannel","Schannel","tw_top","tw_antitop"},
        kMagenta-3, "Single top");
    groups.emplace_back("TTbarSemi",
        vector<string>{"ttbar_SemiLeptonic"},
        kRed, "t#bar{t} semi-lep.");
    groups.emplace_back("TTbarFull",
        vector<string>{"ttbar_FullyLeptonic"},
        kAzure+1, "t#bar{t} fully-lep.");

    // Axis label map
    const map<string,string> axis_labels = AxisLabels();

    // Histograms to plot
    const vector<string> hist_list = {
      "pTSum", "AL", "planarity", "Sxz","Szz","p2in","nJet","delta_R","FW1"
    };

    // Output ROOT file
    gSystem->mkdir("plots", kTRUE);
    TFile* outf = new TFile(("plots/all_plots_BDT_" + ERA + ".root").c_str(), "RECREATE");
    if (!outf || outf->IsZombie())
        cerr << "WARNING: could not create output ROOT file." << endl;

    // Main loop
    for (const string& hname : hist_list)
        PlotHistogram(hname, mc_file, qcd_file, data_file,
                      norm_factors, tf_map, axis_labels,
                      groups, era, ERA, outf);

    if (outf && !outf->IsZombie()) {
        outf->Write();
        outf->Close();
        delete outf;
        cout << "All canvases written to plots/all_plots_BDT_" << ERA << ".root" << endl;
    }

    // Final cleanup
    for (auto& g : groups) { delete g.hist; g.hist = nullptr; }

    cout << "\n" << string(60,'=') << "\nDone. Plots saved in plots/\n"
         << string(60,'=') << endl;
}
