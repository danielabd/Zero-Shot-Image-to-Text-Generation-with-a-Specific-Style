# Pytorch Implementation of Zero-Shot-Image-to-Text-Generation-with-a-Specific-Style

## Approach
![](git_images/Architecture.jpg)


## Example of capabilities
variety successful captions of variety images according to the desired sentiment.
![](git_images/negative_img.png)
![](git_images/positive_img.png)
![](git_images/neutral_img.png)
The Effect of λ_sentiment on the Caption. as λ is bigger the description in the desired
sentiment perspective is stronger but the maintaining on right language attribute is compromised
![](git_images/different_lambda.png)

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
