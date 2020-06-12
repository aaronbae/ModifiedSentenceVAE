# ModifiedSentenceVAE
This project attempts to apply VAE model structure in paraphrase generation task. I used MS Coco and ParaNMT which are relatively short sentences but easy for the models to create embeddings for. Note that this project was in conjunction with the development of GAN and VAE-GAN modesl for paraphrase generation, but this repository only contains my work which was VAE model. Here are some sampled results from my model:
<p align="center">
  <img src="https://github.com/aaronbae/ModifiedSentenceVAE/blob/master/report/samples.PNG" width="700" title="Samples from the Models">
</p>
The samples are coherent, but lacks proximity to the targeted paraphrase. The problem seems to lie in the decoder part of the pipeline. Here is the full report on all 3 models for paraphrase generation:
https://github.com/aaronbae/ModifiedSentenceVAE/blob/master/report/FinalReport.pdf

## Citation
Note that this project was built upon the project Sentence-VAE found in the link below:
- Git: https://github.com/timbmg/Sentence-VAE
- Paper: https://arxiv.org/pdf/1511.06349.pdf
Also, this project was conducted as a part of a team research project with Dheeru Dua and Ananth Gottumukkala.
