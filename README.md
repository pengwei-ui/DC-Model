<h2>DC:Dual-Level Contrastive Learning for Improving Conciseness of Summarization</h2>

This repo is for our paper "Dual-Level Contrastive Learning for Improving Conciseness of Summarization". 



### Results

The following are ROUGE scores calcualted by the standard ROUGE Perl package.

- CNNDM

  |               | ROUGE-1 | ROUGE-2 | ROUGE-L | VAR   |
  | ------------- | ------- | ------- | ------- | ----- |
  | BART          | 44.33   | 21.15   | 41.06   | 0.022 |
  | BRIO          | 47.78   | 23.55   | 44.56   | 0.02  |
  | LFPE          | 45.93   | 22.30   | 42.44   | 0.03  |
  | DC(Our model) | 47.82   | 23.59   | 44.63   | 0.017 |

- XSum

  |         | ROUGE-1 | ROUGE-2 | ROUGE-L | VAR    |
  | ------- | ------- | ------- | ------- | ------ |
  | BART    | 45.14   | 22.27   | 37.25   |        |
  | PEGASUS | 47.38   | 24.54   | 39.41   | 0.0054 |
  | BRIO    | 49.07   | 25.59   | 40.47   | 0.0049 |
  | DC      | 47.75   | 24.86   | 39.72   | 0.0052 |

  

We uploaded the model weights to Google Cloud Drive for your use, [CNNDM](https://drive.google.com/drive/folders/11aOU5Yla5H1NjwQD-n-BrOcs98OfiigX?usp=sharing) and  [XSum](https://drive.google.com/drive/folders/15wN3BuntilDZeKusZoWtfmauzezEHw8b?usp=sharing) weights files, You could load these checkpoints using `model.load_state_dict(torch.load(path_to_checkpoint))`.

