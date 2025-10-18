import pandas as pd
import ast

class PreAndPostProcessCSVFile:
    def __init__(self, input_file, output_file, window=4):
        """
        Args:
            input_file (str): Path to input Excel file
            output_file (str): Path to save processed Excel file
            window (int): Lookahead window for smoothing
        """
        self.input_file = input_file
        self.output_file = output_file
        self.window = window

    def TeamClassifierSmoother(self):
        """Load, smooth, and save team_classifier column in one step."""
        # Load dataframe
        df = pd.read_excel(self.input_file)
        # Ensure team_classifier is dict type
        def parse_dict(val):
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except:
                    return {}
            elif isinstance(val, dict):
                return val
            return {}
        df["team_classifier"] = df["team_classifier"].apply(parse_dict)
        # Apply smoothing
        dicts = df["team_classifier"].tolist()
        n = len(dicts)
        smoothed = []
        for i in range(n):
            current = dicts[i].copy()
            for key, value in current.items():
                if value == 1:
                    future_vals = [
                        dicts[j].get(key, 0)
                        for j in range(i+1, min(i+1+self.window, n))
                    ]
                    if all(v == 0 for v in future_vals):
                        current[key] = 0
            smoothed.append(current)
        df["team_classifier"] = smoothed
        # Save processed file
        df.to_excel(self.output_file, index=False)
        print(f"✅ Processing done, file saved at {self.output_file}")


# ---------------- USAGE ----------------
if __name__ == "__main__":
    input_file = "/media/mtl/Volume F/PROJECTS/projects/Soccer/player_goalkeeper_referee_df.xlsx"
    output_file = "/media/mtl/Volume F/PROJECTS/projects/Soccer/team_classifier_smoothed.xlsx"

    processor = PreAndPostProcessCSVFile(input_file, output_file, window=4)
    processor.TeamClassifierSmoother()
