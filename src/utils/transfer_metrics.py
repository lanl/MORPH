import argparse

def transfer_scores(E_nb, E_pt, E_scratch_full, decimals=4):
    """
    Inputs are MSE scalars:
      E_nb            = naive-baseline error
      E_pt            = pretrained model error (e.g., zero-shot or 1-epoch)
      E_scratch_full  = fully trained scratch error on the target

    Returns a dict with:
      GCR = (E_nb - E_pt) / (E_nb - E_scratch_full)
      OR  = E_pt / E_scratch_full
      NBG = 1 - E_pt / E_nb
    """
    # basic sanity checks (keep or remove if you want it even simpler)
    if E_nb <= 0 or E_scratch_full < 0 or E_pt < 0:
        raise ValueError("Errors must be non-negative and E_nb > 0.")
    if E_nb <= E_scratch_full:
        raise ValueError("Need E_nb > E_scratch_full for ZFG to be defined.")
    
    gcr = (E_nb - E_pt) / (E_nb - E_scratch_full)
    oratio = E_pt / E_scratch_full if E_scratch_full > 0 else float("inf")
    nbg = 1 - (E_pt / E_nb)

    return {
        "GCR": round(gcr, decimals),       # Gap closure ratio (NB to PT vs NB to scratch-full)
        "OR": round(oratio, decimals),     # Optimality Ratio (PT vs scratch-full)
        "NBG": round(nbg, decimals)        # NB Gain (PT vs naive baseline)
    }

# --- example ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate transfer scores for MORPH models")
    parser.add_argument("--E_nb", type=float, required=True, 
                        help="Naive-baseline error")
    parser.add_argument("--E_pt", type=float, required=True, 
                        help="Pretrained model error")
    parser.add_argument("--E_scratch_full", type=float, required=True, 
                        help="Fully trained scratch error on the target")
    args = parser.parse_args()
    scores = transfer_scores(E_nb=args.E_nb, E_pt=args.E_pt, E_scratch_full=args.E_scratch_full)
    print(scores)  
