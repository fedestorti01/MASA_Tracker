import argparse
import os
from plot_metrics import generate_performance_plots_from_csv

def main():
    parser = argparse.ArgumentParser(description="Rigenera grafici da sessione salvata")
    parser.add_argument("-session", type=str, required=True, help="Path della directory di sessione")
    args = parser.parse_args()

    # Determina plots_dir
    plots_dir = os.path.join(args.session, "plots")

    # Genera grafici
    generate_performance_plots_from_csv(
        session_dir=args.session,
        plots_dir=plots_dir,
        save_plots=True
    )


if __name__ == "__main__":
    main()