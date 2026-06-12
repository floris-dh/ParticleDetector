from alpha_spectrum_analyser import AlphaSpectrumAnalyser
from read_and_process import AlphaPulseProcessor

target_source = "Ra226"  
cfg = r"Data\Ra226-050626-2_output\pulse_integrals.csv"

analyser = AlphaSpectrumAnalyser(
    csv_path=cfg,
    isotope_name=target_source,
)

analyser.run_full_analysis()