import os
import pandas as pd

class BaseLogger:
    def __init__(self, log_save_dir, log_name, params=None, decimal_places=2):
        self.log_save_dir = log_save_dir
        self.log_name = log_name
        self.params = params
        self.log_file_loc = os.path.join(log_save_dir, log_name)
        self.decimal_places = decimal_places
        if os.path.isfile(self.log_file_loc):
            self.log_data = pd.read_csv(self.log_file_loc)
        else:
            self.log_data = pd.DataFrame(columns=['PARAM', 'Epoch', 'ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall'])
            self._save_to_csv()

        os.makedirs(log_save_dir, exist_ok=True)

    def close_logger(self):
        self._save_to_csv()
        print(f"Log saved to {self.log_file_loc}")

    def write_parameters(self, parameters):
        print("\nThe parameters: %.3f" % parameters)
        self.params = parameters

    def write_val(self, epoch, loss_tr, scores):
        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.6f\t Acc = %.4f" % (epoch, loss_tr, scores[0]))
        new_row = pd.DataFrame({
            'PARAM': [ self.params],
            'Epoch': [epoch],
            'ACC': [round(scores[0], self.decimal_places)],
            'NMI': [round(scores[1], self.decimal_places)],
            'Purity': [round(scores[2], self.decimal_places)],
            'ARI': [round(scores[3], self.decimal_places)],
            'Fscore': [round(scores[4], self.decimal_places)],
            'Precision': [round(scores[5], self.decimal_places)],
            'Recall': [round(scores[6], self.decimal_places)]
        })
        self.log_data = pd.concat([self.log_data, new_row], ignore_index=True)
        self._save_to_csv()

    def _save_to_csv(self):
        self.log_data.to_csv(self.log_file_loc, index=False)