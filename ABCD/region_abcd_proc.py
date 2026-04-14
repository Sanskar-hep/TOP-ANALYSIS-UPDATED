import hist
import dask
import awkward as ak
import hist.dask as hda
import numpy as np
import dask_awkward as dak
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.lookup_tools import extractor
from coffea.dataset_tools import apply_to_fileset, max_chunks, preprocess
import os
from coffea.nanoevents.methods import vector
import correctionlib
from correctionlib import CorrectionSet
import json  
import warnings
import operator
warnings.filterwarnings("ignore")

class ElectronChannel(processor.ProcessorABC):
    def __init__(self, year = "2017" ,region = "D",btagWP="M",pileUpWP = "L",jetPt=20.0, choice=1):
        
        '''
        Enhanced ElectronChannel processor with two ABCD method choices
        
        Parameters:
        ------------
        year : str
            Data Taking year (2016postVFP,2016preVFP ,2017, 2018)
        
        region : str
            ABCD regions (A, B ,C ,D)

        btag WP : str
            B-tagging working point (L ,M ,T)

        pileUp WP : str
            PileUp jet ID working point (L ,M ,T)

        jet_pt : float
            Minimum pt of the jet required in the analysis

        choice : int 
            ABCD method choice:
            --1. Cutbased Id vs mT (requires >=1 btags for all regions)
            --2. Cutbased Id vs nbtags (requires mT >50 for all regions)
        
        '''
        #-------define all the constants here -------#
        self.region = region
        self.year = year
        self.bTagWP = btagWP
        self.PileUpWP = pileUpWP
        self.choice = choice
        self.JetPt = jetPt
        
        # ----- Validate Inputs ----------#
        self.available_eras = ["2016postVFP", "2016preVFP", "2017" ,"2018"]
        self.available_regions = ["A","B", "C" ,"D"]
        self.available_choices = [1,2]


        if self.year not in self.available_eras:
            raise ValueError(f"Invalid year : {self.year}. Choose from {self.available_eras}")
        if self.region not in self.available_regions:
            raise ValueError(f"Invalid region: {self.region}. Choose from {self.available_regions}")
        if self.choice not in self.available_choices:
            raise ValueError(f"Invalid choice: {self.choice}. Choose from {self.available_choices}")
    
        # ----------CHOICE 1 : CUTBASED ID VS mT (WITH >=1 BTAGS) --------#
        self.region_def_choice1 = {
            "A":{"CutbasedId":1, "wp":"Veto","mT":40,"mT_op":"<","nbtags":1, "nbtags_op":">="},
            "B":{"CutbasedId":4, "wp":"Tight", "mT":40,"mT_op":"<","nbtags":1, "nbtags_op":">="},
            "C":{"CutbasedId":1, "wp":"Veto", "mT":50,"mT_op":">", "nbtags":1, "nbtags_op":">="},
            "D":{"CutbasedId":4, "wp":"Tight","mT":50,"mT_op":">", "nbtags":1, "nbtags_op":">="}
        }
        
        # ----------CHOICE 2 : CUTBASED ID VS NBTAGS (WITH mT > 50)-------#
        self.region_def_choice2 = {
            "A": {"CutbasedId": 1, "wp": "Veto", "mT": 50, "mT_op": ">", "nbtags": 0, "nbtags_op": "=="},
            "B": {"CutbasedId": 4, "wp": "Tight", "mT": 50, "mT_op": ">", "nbtags": 0, "nbtags_op": "=="},
            "C": {"CutbasedId": 1, "wp": "Veto", "mT": 50, "mT_op": ">", "nbtags": 1, "nbtags_op": ">="},
            "D": {"CutbasedId": 4, "wp": "Tight", "mT": 50, "mT_op": ">", "nbtags": 1, "nbtags_op": ">="}
        }
        
        # ----- REGION SELECTION CHOICE -------------------#
        self.RegionConfig = {
            1 : self.region_def_choice1,
            2 : self.region_def_choice2
        }
        
        # ----- DEPLOY THE OPERATION ----------------------#
        self.operations = self.RegionConfig[self.choice][self.region]
        
        self.etaEl_values = {
            "2016postVFP":2.1,
            "2016preVFP": 2.1,
            "2018": 2.4,
            "2017":2.4
        }

        self.HLT_wp = {
            "A":"Veto",
            "B":"Tight",
            "C":"Veto",
            "D":"Tight"
        }
        
        self.HLT_scalefactor = self.HLT_wp[self.region]
        #------for electron--------------#
        self.ElectronPt = 37
        self.ElectronEta = self.etaEl_values[self.year]
        self.CutBasedIdTight = self.operations["CutbasedId"]
        self.IdWP = self.operations["wp"]
        
        #------Barrel and End Cap regions----# 
        self.barrel = 1.444
        self.endcap = 1.566
        
        #------for Jets----------------------#
        self.JetEta = 2.4  
        
        #self.bTagWP = "M"
        self.btag_thresholds_by_year = {
            # Values from: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation
            "2018": {
                "L": 0.0490,
                "M": 0.2783,
                "T": 0.7100
            },
            "2017": {
                "L": 0.0532,
                "M": 0.3040,
                "T": 0.7476
            },
            "2016postVFP": {
                "L": 0.0480,
                "M": 0.2489,
                "T": 0.6377
            },
            "2016preVFP": {
                "L": 0.0508,
                "M": 0.2598,
                "T": 0.6502
            }
        }

        self.DeepJetWP = self.btag_thresholds_by_year[self.year][self.bTagWP]
        
        #-----mT mask------------------------#

        self.mT = self.operations["mT"]
        self.mT_operator = self.operations["mT_op"]
        self.nbtags_threshold = self.operations["nbtags"]
        self.nbtags_operator = self.operations["nbtags_op"]

        #-----Operations on mT(if choosen option =1) and nbtags(if choosen option = 2)--------------#
        self.ops = {
            ">":operator.gt,
            "<":operator.lt,
            ">=":operator.ge,
            "<=":operator.le,
            "==":operator.eq
        }

        #-----Event Selection Requirements------#
        self.nElectron = 0   
        self.nJets = 4
                
        #------Flavor specific Jets----------#
        self.bflav = 5
        self.cflav = 4
        self.lflav = 0
        
        #-------Histogram for Electron-------#
        #----For PT---#
        self.ePtBins = 27
        self.ePtStart = 30
        self.ePtEnd = 300
        
        #---For Eta----#
        self.etaConfig = {
            "2016postVFP":42,
            "2016preVFP":42,
            "2018":48,
            "2017":48
        }


        self.eEtaBins = self.etaConfig[self.year]
        self.eEtaStart= -self.ElectronEta
        self.eEtaEnd = self.ElectronEta
        
        #----For Phi---#
        self.ePhiBins = 20
        self.ePhiStart = -np.pi
        self.ePhiEnd = np.pi
        
        #------Histogram for Jet-------------#
        #----FOR PT--------#
        self.jPtBins = 38
        self.jPtStart= 20
        self.jPtEnd = 400
        
        #----FOR ETA-------#
        self.jEtaBins = 48
        self.jEtaStart = -2.4
        self.jEtaEnd = 2.4
        
        #----FOR PHI-------#
        self.jPhiBins = 20
        self.jPhiStart = -np.pi
        self.jPhiEnd = np.pi
        
        #---Pileup Jet id wp---#

        self.PileUpConfig = {
            "2016postVFP":{
                "L": (1 << 0),
                "M": (1 << 1),
                "T": (1 << 2)
            },
            "2016preVFP":{
                "L": (1 << 0),
                "M": (1 << 1),
                "T": (1 << 2)
            },
            "2017":{
                "L": (1 << 2),
                "M": (1 << 1),
                "T": (1 << 0)
            },
            "2018":{
                "L": (1 << 2),
                "M": (1 << 1),
                "T": (1 << 0)
            }
        
        }


    def process(self, events):
        '''
        Process events and apply ABCD method based on selected choice
        '''
        isRealData = not hasattr(events, "GenPart")
        dataset = events.metadata['dataset']
        print("\n")
        print(f"current dataset : {dataset}")
        print("--" * 50)
        
        
        selection = PackedSelection()
        
        #----Print the characteristics of the current region -----#
        print("\n" + "="*60)
        print("ABCD METHOD CONFIGURATION")
        print("="*60)
        print(f"Choice               : {self.choice}")
        if self.choice == 1:
            print(f"Method               : CutBasedID vs mT")
            print(f"B-tag requirement    : >= 1 for all regions")
            print(f"================FLOWCHART=========================")
            print("                      ID =Loose       ID = Tight")
            print("                  ┌─────────────────┬─────────────────┐")
            print("    mT < 40       │   Region A      │   Region B      │")
            print("                  │  (nBtags >= 1)  │  (nBtags >= 1)  │")
            print("                  ├─────────────────┼─────────────────┤")
            print("    mT > 50       │   Region C      │   Region D      │")
            print("                  │  (nBtags >= 1)  │  (nBtags >= 1)  │")
            print("                  └─────────────────┴─────────────────┘")
        else:
            print(f"Method               : CutBasedID vs nBtags")
            print(f"mT requirement       : > 50 for all regions")
            print(f"================FLOWCHART=========================")
            print("                      ID =Loose       ID = Tight")
            print("                  ┌─────────────────┬─────────────────┐")
            print("    nBtags = 0    │   Region A      │   Region B      │")
            print("                  │  (mT > 50)      │  (mT > 50)      │")
            print("                  ├─────────────────┼─────────────────┤")
            print("    nBtags >=1    │   Region C      │   Region D      │")
            print("                  │  (mT > 50)      │  (mT > 50)      │")
            print("                  └─────────────────┴─────────────────┘")

        
        print(f"Region               : {self.region}")
        print(f"Year                 : {self.year}")
        print(f"Jet pT threshold     : {self.JetPt} GeV")
        print(f"B-tag WP             : {self.bTagWP} ({self.DeepJetWP:.4f})")
        print(f"CutBased ID          : {self.IdWP} ({self.CutBasedIdTight})")
        print(f"mT cut               : {self.mT_operator} {self.mT} GeV")
        print(f"nBtags cut           : {self.nbtags_operator} {self.nbtags_threshold}")
        print("="*60 + "\n")

        #------Check for events.HLT.fields for correct trigger -------#
        def get_trigger_mask(hlt ,era):
            if era in ["2016postVFP","2016preVFP"]:
                return hlt.Ele32_eta2p1_WPTight_Gsf

            elif era in ["2018"]:
                return hlt.Ele32_WPTight_Gsf

            elif era in ["2017"]:
                return hlt.Ele35_WPTight_Gsf

            else:
                raise ValueError(f"The era you are trying to analyse is not present ! please pick among {self.available_eras}")

        #------Check the official twiki page for Pileup jet Ids for RUN2 UL------#
        def get_puId_mask(jets ,era):
            if era in self.available_eras:
                puId_mask = (jets.puId & self.PileUpConfig[era][self.PileUpWP])!=0
                return puId_mask

            else : 
                raise ValueError(f"The era you are trying to analyse is not present ! please pick among {self.available_eras}")
        
        #------For Good Jets ---------- #
        def get_good_jets(jets):
            has_tight_Id = (jets.jetId & (1 << 2))!= 0
            
            #----apply puId only for jets with pt < 50 ------#
            has_puId = get_puId_mask(jets,self.year)
            passes_puId = dak.where(jets.pt > 50 , True , has_puId)
            
            is_jet = (jets.pt >= self.JetPt) & (abs(jets.eta) < self.JetEta) & has_tight_Id & passes_puId 
            
            return is_jet
        
        #------For Good Electrons ------#
        def get_good_electrons(electrons):
            basic_cuts = (
                (electrons.pt>=self.ElectronPt)& (abs(electrons.eta) < self.ElectronEta) & ~((abs(electrons.eta) >= self.barrel) & (abs(electrons.eta) <= self.endcap)) & (electrons.cutBased == self.CutBasedIdTight))
            
            is_barrel = abs(electrons.eta) < self.barrel
            
            max_dxy = dak.where(is_barrel,0.05,0.10)
            max_dz = dak.where(is_barrel,0.10,0.20)
            
            ip_cuts = (abs(electrons.dxy) <= max_dxy) & (abs(electrons.dz) <= max_dz)
            
            return basic_cuts & ip_cuts
        
        # ------- Select Good Objects --------------------------------#
        good_electrons = events.Electron[get_good_electrons(events.Electron)]
        leading_electron = dak.firsts(good_electrons)
        met = events.MET
        
        # ---------Calculate Transverse Mass -------------------------#
        dphi = leading_electron.phi - met.phi
        dphi = (dphi + np.pi)%(2*np.pi) - np.pi
        mT = np.sqrt(2 * leading_electron.pt * met.pt * (1 - np.cos(dphi)))
        mT_mask = self.ops[self.mT_operator](mT, self.mT)
        mT_mask_filtered = dak.fill_none(mT_mask , False)

        # ------- B-Tagging Selection --------------------------------#
        is_btagged = (events.Jet.btagDeepFlavB > self.DeepJetWP) & get_good_jets(events.Jet)
        nbtags_num = dak.sum(is_btagged, axis=1)
        nbtags_mask = self.ops[self.nbtags_operator](nbtags_num, self.nbtags_threshold)

        # ------- Build Selection ------------------------------------#
        selection.add_multiple({
            "Trigger": get_trigger_mask(events.HLT,self.year),
            "good_electrons": dak.sum(get_good_electrons(events.Electron),axis=1) > self.nElectron , 
            "has_atleast_4jets":dak.sum(get_good_jets(events.Jet), axis=1) >= self.nJets,
            "nbtags_criteria":nbtags_mask,
            "mT_criteria":mT_mask_filtered
        })
        
        # --------Apply Full Selection---------------------------------#
        mask = selection.all("Trigger", "good_electrons", "has_atleast_4jets","nbtags_criteria","mT_criteria")
        
        # --------Select Objects After All Cuts -----------------------#
        selected_jets = events.Jet[mask]
        sjets = selected_jets[get_good_jets(selected_jets)]
        
        ele = events.Electron[mask]
        sel = dak.firsts(ele[get_good_electrons(ele)])
        
        #-------sorting sjets based on their pt-------------# 
        sjet_sorted = dak.argsort(sjets.pt , ascending=False)
        sjets = sjets[sjet_sorted]
        
        if isRealData :
            sumw = dak.num(events ,axis=0)            
        else :
            sumw = dak.sum(events.Generator.weight)
           
            
        nselected = dak.sum(mask)
        #print(nselected.compute()) 
        
        #------- Cutflow and n-1 selections--------# 
        cutflow = selection.cutflow(*selection.names)
        honecut, hcutflow, labels = cutflow.yieldhist()
        
        nminusone = selection.nminusone(*selection.names)
        hnminusone, nlabels = nminusone.yieldhist()
        #-------------------------------------------#
        
        #------ Weights--------#
        weights = Weights(size= None, storeIndividual=True)
        
        if isRealData:
            print(f"RealData block is being executed for {dataset}.....")
            if hasattr(events, "L1PreFiringWeight"):
                print("PrefiringWeight is present")
                weights.add("PreFiringWeight",weight=events[mask].L1PreFiringWeight.Nom, weightUp=events[mask].L1PreFiringWeight.Up, weightDown=events[mask].L1PreFiringWeight.Dn)
               
            else:
                print("Prefiring weight is not present")
                weights.add("PreFiringWeight", weight=dak.ones_like(sel.pt))
            
            print(list(weights._weights.keys()))    
        else:
            print(f"MC block is being executed for {dataset}.....")
            if hasattr(events, "puWeight"):
                print("pileup weight is present")
                weights.add("pileupWeight",weight=events[mask].puWeight, weightUp=events[mask].puWeightUp, weightDown=events[mask].puWeightDown)
            else:
                print("pileup weight is not present")
                weights.add("pileupWeight", weight=dak.ones_like(sel.pt))
                
            if hasattr(events, "L1PreFiringWeight"):
                print("L1prefiring weight is present")
                weights.add("PreFiringWeight",weight=events[mask].L1PreFiringWeight.Nom, weightUp=events[mask].L1PreFiringWeight.Up, weightDown=events[mask].L1PreFiringWeight.Dn)
            else:
                print("Prefiring weight is not present")
                weights.add("PreFiringWeight", weight=dak.ones_like(sel.pt))
                
            if hasattr(events, "LHEWeight"):
                print("LHEWeight is present")
                weights.add("LHEWeight",weight=events[mask].LHEWeight.originalXWGTUP / abs(events[mask].LHEWeight.originalXWGTUP))
            else:
                print("LHEWeight is not present")
                weights.add("LHEWeight", weight=dak.ones_like(sel.pt))
                
            #---------location of the root and the json files------#
            Dir = "/nfs/home/sanskar/SF/SFs"
            Dir1 = f"/nfs/home/sanskar/SF/SFs/{self.year}/WITH_NBTAGS_VS_ID"
            filename1 = f"UL{self.year}_el_HLT_{self.HLT_scalefactor}.root"
            filename2 = f"UL{self.year}_el_ID.json"
            filename3 = f"UL{self.year}_jet_Btagging.json"
            filename4 = f"btag_eff_region{self.region}_{self.year}_correctionlib_with_id_and_nbtags.json"
            filename5 = f"UL{self.year}_jet_jmar.json"
            
            filename3a = "UL2016preVFP_jet_Btagging.json"
            #----------Concatenate File Paths----------------------#  
            full_path1 = os.path.join(Dir, filename1) 
            full_path2 = os.path.join(Dir, filename2)  
            full_path3 = os.path.join(Dir ,filename3)
            full_path4 = os.path.join(Dir1 ,filename4)
            full_path5 = os.path.join(Dir ,filename5)
            
            full_path3a = os.path.join(Dir,filename3a)
            #------------HLT sf---------------------#
            if os.path.isfile(full_path1):
                ext = extractor()
                ext.add_weight_sets([f"* * {full_path1}"])
                ext.finalize()
                evaluator = ext.make_evaluator()
                electron_sf = evaluator["EGamma_SF2D"](sel.eta, sel.pt)
                weights.add("HLT_SF", weight=electron_sf)
                print("HLT scale factors :applied ")
            else:
                print("HLT scale_factors: not present adding +1 to weight")
                weights.add("HLT_SF", weight=dak.ones_like(sel.pt))
                
            #-----------electron id sf--------------#
            if os.path.isfile(full_path2):
                eset = CorrectionSet.from_file(full_path2)
                ele_corr = eset["UL-Electron-ID-SF"]
                
                reco_sf = ele_corr.evaluate(self.year, "sf", "RecoAbove20", sel.eta, sel.pt)
                reco_sf_up = ele_corr.evaluate(self.year, "sfup", "RecoAbove20", sel.eta, sel.pt)
                reco_sf_down = ele_corr.evaluate(self.year, "sfdown", "RecoAbove20", sel.eta, sel.pt)
                
                id_sf = ele_corr.evaluate(self.year, "sf", self.IdWP, sel.eta, sel.pt)
                id_sf_up = ele_corr.evaluate(self.year, "sfup", self.IdWP, sel.eta, sel.pt)
                id_sf_down = ele_corr.evaluate(self.year, "sfdown", self.IdWP, sel.eta, sel.pt)
                
                ele_id_sf = reco_sf * id_sf
                ele_id_sf_up = reco_sf_up * id_sf_up
                ele_id_sf_down = reco_sf_down * id_sf_down
                
                weights.add("eleID", weight=ele_id_sf, weightUp=ele_id_sf_up, weightDown=ele_id_sf_down)
                print("ele_id : applied succesfully")
                
            else :
                print("ele_id :not present , adding +1 to weights")
                weights.add("eleID" , weight = dak.ones_like(sel.pt))
                
                
            #--------------btag sf------------------# 
            if os.path.isfile(full_path3):
                # load the b-tagging correction
                cset= CorrectionSet.from_file(full_path3)
                btag_corr_heavy = cset["deepJet_comb"]

                if self.year == "2016postVFP":
                    cset_preVFP = CorrectionSet.from_file(full_path3a)
                    btag_corr_light = cset_preVFP["deepJet_incl"]
                else:
                    btag_corr_light = cset["deepJet_incl"]

                #---Load the b-tagging efficiencies----
                bset = CorrectionSet.from_file(full_path4)
                btag_eff = bset[f"btag_eff_region{self.region}_DeepJet_Medium"]
                
                is_light = sjets.hadronFlavour == self.lflav
                is_heavy = (sjets.hadronFlavour == self.bflav)|(sjets.hadronFlavour == self.cflav)
                
                flavor_for_heavy = dak.where(is_heavy , sjets.hadronFlavour ,self.bflav)
                flavor_for_light = dak.where(is_light , sjets.hadronFlavour ,self.lflav)
                
                jet_eff = btag_eff.evaluate(dataset , sjets.hadronFlavour , abs(sjets.eta) ,sjets.pt)
                
                tagged = sjets.btagDeepFlavB > self.DeepJetWP
                
                p_mc = dak.where(tagged , jet_eff , 1-jet_eff)
                p_mc_event = dak.prod(p_mc ,axis=1)
                
                event_weight = {}
                for variation in ["central","up","down"]:
                    sf_heavy = btag_corr_heavy.evaluate(variation, self.bTagWP, flavor_for_heavy, abs(sjets.eta), sjets.pt)
                    sf_light = btag_corr_light.evaluate(variation, self.bTagWP, flavor_for_light, abs(sjets.eta), sjets.pt)
                    
                    jet_sf = dak.where(is_heavy, sf_heavy ,sf_light)
                    
                    eff_data = jet_eff * jet_sf
                    
                    p_data = dak.where(tagged, eff_data ,1-eff_data)
                    p_data_event = dak.prod(p_data ,axis=1)
                    
                    event_weight[variation] = dak.where(p_mc_event >0 , p_data_event/p_mc_event ,1.0)
                    
                weights.add("btag_sf",weight = event_weight["central"], weightUp = event_weight["up"] ,weightDown = event_weight["down"])
                print("Btagging Corrections : APPLIED")
            else:
                print("Btagging Corrections : NOT APPLIED")
                weights.add("btag_sf", weight = dak.ones_like(sel.pt))
            
            
            #-------PIle up jet id corrections-----#
            if os.path.isfile(full_path5):
                pset = CorrectionSet.from_file(full_path5)
                pjetid_corr= pset["PUJetID_eff"]
                
                puId_jet_pt_mask = sjets.pt <= 50
                
                dummy_pt = dak.where(sjets.pt > 50, 50,sjets.pt)
                puId_weight = {}
                
                for variation in ["nom","up","down"]:
                    sf_puId = pjetid_corr.evaluate(sjets.eta ,dummy_pt ,variation , self.PileUpWP)
                    sf = dak.where(puId_jet_pt_mask , sf_puId ,1.0)
                    
                    sf_multiplied = dak.prod(sf, axis=1)
                    
                    puId_weight[variation] = sf_multiplied
                
                weights.add("pileUp_sf",weight = puId_weight["nom"], weightUp=puId_weight["up"],weightDown=puId_weight["down"])
                print("PILE-UP JETID CORRECTIONS: APPLIED")
            else:
                print("PILEUP JET-ID CORRECTIONS: NOT APPLIED")
                weights.add("pileUp_sf",weight = dak.ones_like(sel.pt))
            

            
            print(list(weights._weights.keys()))
            
        #---------------------------------------- HISTOGRAMS ------------------------------------------#
        electron_pt = (
            hda.Hist.new
            .Reg(self.ePtBins, self.ePtStart, self.ePtEnd, name="pt", label="$p_T$")
            .Weight()
        )
        
        electron_eta = (
            hda.Hist.new
            .Reg(self.eEtaBins, self.eEtaStart, self.eEtaEnd, name="eta", label="$\\eta$")
            .Weight()
        )
        electron_phi = (
            hda.Hist.new
            .Reg(self.ePhiBins, self.ePhiStart , self.ePhiEnd ,name ="phi",label= "$\\phi$")
            .Weight()
        
        )
        
        Jet_pt = (
            hda.Hist.new
            .Reg(self.jPtBins,self.jPtStart,self.jPtEnd, name="jet_pt" , label ="Jet $p_T$")
            .Weight()
        )
        
        Jet_eta = (
            hda.Hist.new
            .Reg(self.jEtaBins, self.jEtaStart, self.jEtaEnd, name = "jet_eta", label="Jet $\\eta$")
            .Weight()
        )
        Jet_phi = (
            hda.Hist.new
            .Reg(self.jPhiBins, self.jPhiStart ,self.jPhiEnd ,name = "jet_phi", label = "Jet $\\phi$")
            .Weight()
        )
        
        
        leading_jet = dak.firsts(sjets)
        
        electron_pt.fill(pt=sel.pt, weight=weights.weight())
        electron_eta.fill(eta=sel.eta, weight=weights.weight())
        electron_phi.fill(phi = sel.phi, weight=weights.weight())
        
        Jet_pt.fill(jet_pt=leading_jet.pt, weight=weights.weight())
        Jet_eta.fill(jet_eta=leading_jet.eta, weight=weights.weight())
        Jet_phi.fill(jet_phi=leading_jet.phi ,weight=weights.weight())
        
        #----------------Return Computed Results----------------------#
        return {
            dataset:{
                "selEvents": nselected,
                "electron_pt":electron_pt,
                "electron_eta":electron_eta,
                "electron_phi":electron_phi,
                "Jet_pt":Jet_pt,
                "Jet_eta":Jet_eta,
                "Jet_phi":Jet_phi
            }
        }
        
    def postprocess(self, accumulator):
        return accumulator
