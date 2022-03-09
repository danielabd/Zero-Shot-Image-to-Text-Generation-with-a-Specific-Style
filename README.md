# Pytorch Implementation of Zero-Shot-Image-to-Text-Generation-with-a-Specific-Style

## Approach
![](git_images/Architecture.jpg)


## Example of capabilities

variety successful captions of variety images according to the desired sentiment:
![](git_images/3_examples_imgs.png)  
  
The Effect of λsentiment on the Caption:  
![](git_images/small_different_lambda.png)  
as λ is larger the description in the desired sentiment perspective is stronger but the coherence of the generated sentence is decreased.

## Usage

### Set up environment:
```bash
$ ./setup.sh
$ conda activate zeroshot
```


### Run model:
```bash
$ python run.py --reset_context_delta
```
### Results: 
See results in results.csv
