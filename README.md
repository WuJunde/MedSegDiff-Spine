

# MedSegDiff: Medical Image Segmentation with Diffusion Model
MedSegDiff a Diffusion Probabilistic Model (DPM) based framework for Medical Image Segmentation. The algorithm is elaborated in our paper [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/abs/2211.00611) and [MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer](https://arxiv.org/pdf/2301.11798.pdf).

<img align="left" width="170" height="170" src="https://github.com/WuJunde/MedSegDiff/blob/master/medsegdiff_showcase.gif"> Diffusion Models work by destroying training data through the successive addition of Gaussian noise, and then learning to recover the data by reversing this noising process. After training, we can use the Diffusion Model to generate data by simply passing randomly sampled noise through the learned denoising process.In this project, we extend this idea to medical image segmentation. We utilize the original image as a condition and generate multiple segmentation maps from random noises, then perform ensembling on them to obtain the final result. This approach captures the uncertainty in medical images and outperforms previous methods on several benchmarks.


## A Quick Overview 

|<img align="left" width="480" height="170" src="https://github.com/WuJunde/MedSegDiff/blob/master/framework.png">|<img align="right" width="450" height="270" src="https://github.com/WuJunde/MedSegDiff/blob/master/frameworkv2.png">|
|:--:|:--:| 
| **MedSegDiff-V1** | **MedSegDiff-V2** |

## Requirement

``pip install -r requirement.txt``

## Example Cases

    
For training, run: ``python scripts/segmentation_train.py --data_dir (where you put data folder)/data/training --out_dir output data direction --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``

For sampling, run: ``python scripts/segmentation_sample.py --data_dir (where you put data folder)/data/testing --out_dir output data direction --model_path saved model --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``


### Run on  your own dataset
It is simple to run MedSegDiff on the other datasets. Just write another data loader file following `` ./guided_diffusion/isicloader.py`` or `` ./guided_diffusion/bratsloader.py``. 

## Suggestions for Hyperparameters and Training
To train a fine model, i.e., MedSegDiff-B in the paper, set the model hyperparameters as:
~~~
--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 
~~~
diffusion hyperparameters as:
~~~
--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
~~~
To speed up the sampling:
~~~
--diffusion_steps 50 --dpm_solver True 
~~~
run on multiple GPUs:
~~~
--multi-gpu 0,1,2 (for example)
~~~
training hyperparameters as:
~~~
--lr 5e-5 --batch_size 8
~~~
and set ``--num_ensemble 5`` in sampling.

Run about 100,000 steps in training will be converged on most of the datasets.

A setting to unleash all its potential is (MedSegDiff++):
~~~
--image_size 256 --num_channels 512 --class_cond False --num_res_blocks 12 --num_heads 8 --learn_sigma True --use_scale_shift_norm True --attention_resolutions 24 
~~~
Then train it with batch size ``--batch_size 64`` and sample it with ensemble number ``--num_ensemble 25``.

## Cite
Please cite
~~~
@article{wu2022medsegdiff,
  title={MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model},
  author={Wu, Junde and Fang, Huihui and Zhang, Yu and Yang, Yehui and Xu, Yanwu},
  journal={arXiv preprint arXiv:2211.00611},
  year={2022}
}
~~~

~~~
@article{wu2023medsegdiff,
  title={MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer},
  author={Wu, Junde and Fu, Rao and Fang, Huihui and Zhang, Yu and Xu, Yanwu},
  journal={arXiv preprint arXiv:2301.11798},
  year={2023}
}
~~~


