

class Configs():
    """Class containing all the experiment configurations saved after hyperparameter tuning"""
    
    def __init__(self):
        self.experiments = dict()
        self.known_experiments = {"MLP",
                                  "GRU",
                                  "GRUAtt",
                                  "GraphRNN",
                                  "GraphRNNAtt",
                                  "GGT",
                                  "GGTCNNAtt",
                                  }
        
        self.experiments["MLP"] = {"batch_size": 64,
                                   "lr_rate": 3e-4,
                                   "weight_decay": 5e-5,
                                   "decoder": "DecoderMLP",
                                   "encoder": "EncoderCNN",
                                   "N": 0,
                                   "n_heads": 0,
                                   "hidden_size": 0,
                                   }
        
        self.experiments["GRU"] = {"batch_size": 16,
                                   "lr_rate": 5e-4,
                                   "weight_decay": 1e-5,
                                   "decoder": "DecoderGRU",
                                   "encoder": "EncoderCNN",
                                   "N": 0,
                                   "n_heads": 0,
                                   "hidden_size": 0,
                                   }
        
        self.experiments["GRUAtt"] = {"batch_size": 16,
                                      "lr_rate": 5e-4,
                                      "weight_decay": 1e-5,
                                      "decoder": "DecoderGRUAtt",
                                      "encoder": "EncoderCNN",
                                      "N": 0,
                                      "n_heads": 0,
                                      "hidden_size": 0,
                                      }
        
        self.experiments["GraphRNN"] = {"batch_size": 64,
                                        "lr_rate": 5e-4,
                                        "weight_decay": 5e-5,
                                        "decoder": "DecoderGraphRNN",
                                        "encoder": "EncoderCNN",
                                        "N": 0,
                                        "n_heads": 0,
                                        "hidden_size": 16,
                                        }
        
        self.experiments["GraphRNNAtt"] = {"batch_size": 64,
                                           "lr_rate": 5e-4,
                                           "weight_decay": 5e-5,
                                           "decoder": "DecoderGraphRNNAtt",
                                           "encoder": "EncoderCNN",
                                           "N": 0,
                                           "n_heads": 0,
                                           "hidden_size": 16,
                                           }
        
        self.experiments["GGT"] = {"batch_size": 64,
                                     "lr_rate": 3e-4,
                                     "weight_decay": 1e-5,
                                     "decoder": "DecoderGGT",
                                     "encoder": "EncoderCNN",
                                     "N": 12,
                                     "n_heads": 8,
                                     "hidden_size": 0,
                                     }
        
        self.experiments["GGTCNNAtt"] = {"batch_size": 64,
                                           "lr_rate": 3e-4,
                                           "weight_decay": 1e-5,
                                           "decoder": "DecoderGGT",
                                           "encoder": "EncoderCNNAtt",
                                           "N": 12,
                                           "n_heads": 8,
                                           "hidden_size": 0,
                                           }
    
    def load_experiment(self, args):
        if args.experiment not in self.known_experiments:
            return args
        else:
            exp = self.experiments[args.experiment]
            args.batch_size = exp["batch_size"]
            args.lr_rate = exp["lr_rate"]
            args.weight_decay = exp["weight_decay"]
            args.decoder = exp["decoder"]
            args.encoder = exp["encoder"]
            args.N = exp["N"]
            args.n_heads = exp["n_heads"]
            args.hidden_size = exp["hidden_size"]
            return args
