# [Frequency Supervised MR-to-CT Image Synthesis, MICCAI workshop on Deep Generative Models, 2021](https://arxiv.org/pdf/2107.08962.pdf)
![image](https://github.com/shizenglin/Frequency-Supervised-MR-to-CT-Image-Synthesis/blob/main/overview.png)
<p> &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 Overview of our approach </p>

<h2> Requirements </h2>
     1. CUDA 8.0 and Cudnn 7.5 or higher
<br> 2. GPU memory 10GB or higher
<br> 3. Python 2.7 or higher 
<br> 4. Tensorflow 2.0 or higher

<h2> Training </h2>
     1. Prepare your data following Section 3.1.
<br> 2. Set the experiment settings in ¨tr_param.ini¨ in which phase = train, and set other parameters accordingly (refer to our paper).
<br> 3. Run ¨python main.py¨

<h2> Testing </h2>
     1. Prepare your data following Section 3.1.
<br> 2. Set the experiment settings in ¨tr_param.ini¨ in which phase = test, and set other parameters accordingly (refer to our paper).
<br> 3. Run ¨python main.py¨


Please cite our paper when you use this code.

     @inproceedings{shi2021frequency,
     title={Frequency-Supervised MR-to-CT Image Synthesis},
     author={Zenglin Shi, Pascal Mettes, Guoyan Zheng and Cees G. M. Snoek},
     booktitle={MICCAI workshop on Deep Generative Models},
     year={2021}
     }


