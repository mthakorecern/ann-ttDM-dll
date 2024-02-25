import subprocess

def evaluate_multiple(input_files, output_files, config_file):
    for input_name, input_path in input_files.items():
        output_file = output_files.get(input_name, "output.root")
        command = ["python3", "evaluate.py", input_path, output_file, config_file]
        subprocess.run(command)

if __name__ == "__main__":
    signal_paths = {
        "DMScalar_top_tChan_Mchi1_Mphi100": "/hdfs/store/user/vshang/tDM_Run2018/DMScalar_top_tChan_Mchi1_Mphi100_TuneCP5_13TeV-madgraph-mcatnlo-pythia8/ModuleCommonSkim_12242022/tree_all.root",
        #"DMScalar_top_tChan_Mchi1_Mphi150": "/hdfs/store/user/vshang/tDM_Run2018/DMScalar_top_tChan_Mchi1_Mphi150_TuneCP5_13TeV-madgraph-mcatnlo-pythia8/ModuleCommonSkim_12242022/tree_all.root",
        # Add other signal paths here
    }

    background_paths = {
        "TTToSemiLeptonic": "/hdfs/store/user/vshang/ttbarPlusJets_Run2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/ModuleCommonSkim_12242022/tree_all.root",
        "TTTo2L2Nu": "/hdfs/store/user/vshang/ttbarPlusJets_Run2018/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/ModuleCommonSkim_12242022/tree_all.root",
        # Add other background paths here
    }

    output_files = {
        "DMScalar_top_tChan_Mchi1_Mphi100": "output_signal1.root",
        #"DMScalar_top_tChan_Mchi1_Mphi150": "output_signal2.root",
        # Add other output file names for signal paths here
        "TTToSemiLeptonic": "output_background1.root",
        "TTTo2L2Nu": "output_background2.root",
        # Add other output file names for background paths here
    }

    config_file = "conf.json"

    evaluate_multiple(signal_paths, output_files, config_file)
    evaluate_multiple(background_paths, output_files, config_file)
