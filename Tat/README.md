# Model training
Please see the template data TatOnlyProt.fa (the protein sequence) and goldenStandarddataIncHet.csv (the file containing all information about mutations).

```
usage: train.py [-h] [-s S] [-n N] [-m M] [-a A] [-p P] [-v V] [-bt BT] [-bc BC] [-epoch EPOCH] [-gpu GPU] [-lr1 LR1] [-lr2 LR2] [-batch BATCH] [-save SAVE]

optional arguments:
  -h, --help    show this help message and exit
  
  -s S          The file containing the protein sequence. (default: None)
  
  -n N          The name of the protein sequence in the file. (default: None)
  
  -m M          The file containing the mutations, the position of the mutations and the output values. (default: None)
  
  -a A          Row name of the activity. (default: Percentage of Reads GFP High over High and Low)
  
  -p P          Row name of the position. (default: Position)
  
  -v V          Row name of the Variant ID. (default: #Variant ID)
  
  -bt BT        Barcode Count threshold. (default: 5)
  
  -bc BC        Row name of the Barcode Count. (default: None)
  
  -epoch EPOCH  The number of epochs. (default: 200)
  
  -gpu GPU      The device to use to train the model. (default: 0)
  
  -lr1 LR1      The learning rate of the task specific layer. (default: 0.001)
  
  -lr2 LR2      The learning rate of the shared task. (default: 1e-05)
  
  -batch BATCH  The batch size. (default: 8)
  
  -save SAVE    The save folder. (default: ./Rep2Mut.p)
  ```

Example

```python train.py -s data/TatOnlyProt.fa -n Tat -m goldenStandarddataIncHet.csv```