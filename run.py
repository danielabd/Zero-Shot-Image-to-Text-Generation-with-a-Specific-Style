import argparse
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_path", type=str, default='example_images/captions/COCO_val2014_000000008775.jpg',
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])

    args = parser.parse_args()

    return args

def run(args, img_path,sentiment_type,log_file):
    text_generator = CLIPTextGenerator(log_file,**vars(args))

    image_features = text_generator.get_img_feature([img_path], None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size,sentiment_type=sentiment_type)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])
    with open(log_file,'a') as fp:
        for c in captions:
            fp.write(c)
        fp.write('best clip:'+args.cond_text + captions[best_clip_idx])

def run_arithmetic(args, imgs_path, img_weights):
    text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])

if __name__ == "__main__":
    args = get_args()
    log_file = 'daniela_log.txt'
    #img_path_list = range(2,13)#daniela ad  option for list of imgs
    # img_path_list = [6,5,11,9]
    img_path_list = [1]
    sentiment_list = ['negative','positive','neutral']
    for i in img_path_list:
        for sentiment_type in sentiment_list:
            args.caption_img_path = "imgs/img"+str(i)+".png" #img_path_list[i]
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f'~~~~~~~~\n{dt_string} | Work on img path: {args.caption_img_path} with ***{sentiment_type}***  sentiment.\n~~~~~~~~')
            with open(log_file,'a') as fp:
                fp.write(f'~~~~~~~~\n{dt_string} | Work on img path: {args.caption_img_path} with ***{sentiment_type}***  sentiment.\n~~~~~~~~')

            if args.run_type == 'caption':
                run(args, img_path=args.caption_img_path,sentiment_type=sentiment_type,log_file=log_file)
            elif args.run_type == 'arithmetics':
                args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
                run_arithmetic(args, imgs_path=args.arithmetics_imgs, img_weights=args.arithmetics_weights)
            else:
                raise Exception('run_type must be caption or arithmetics!')