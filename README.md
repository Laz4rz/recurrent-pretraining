# Pretraining a Depth-Recurrent Model

This repo contains the code we used to train a recurrent-depth model at scale on 4096 AMD GPUs on Frontier. All details on this model can be found in the tech report: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (https://www.arxiv.org/abs/2502.05171). The final model is `huginn-0125`, which can be found here: https://huggingface.co/tomg-group-umd/huginn-0125. 

This repo is based on a fork of https://github.com/Lightning-AI/litgpt, which was very helpful to bootstrap our efforts, but little `litgpt` code remains at this stage. Code in this repository was written by Jonas Geiping, John Kirchenbauer, Sean McLeish, Khalid Saifullah, Manli Shu, Neel Jain, Siddarth Singh, Abhimanyu Hans, Monte Hoover and Prajwal Singhanaia.

This repo also contains all code to prepare the tokenizer and data, mostly in `scripts/`. 

I (Jonas) do not necessarily think that you should pretrain your own model with this implementation, but I hope it serves as a useful reference for the exact choices we took to run this model (at all), and how we ran this model given the limitations of AMD systems. **If you are working with either of these, feel free to always raise an issue asking for more details.**


## Code Setup:
*  The actual model definition is in `repre/model_dynamic.py`.
*  The training is orchestrated from `train.py`.
*  Model shapes can be found in `recpre/model_registry.py`. The final model is the shape `nebel-raven-3.5b`
*  The configurations for our two large-scale runs are in `launch_configs/`. 
*  The environment flags can be read out of `launch_frontier.py`.
* The parallelism implementation is deep down in `recpre/utils.py`, in a class called `SimpleFabric`. `_allreduce_chunk_stream` was used for inter-node communication, which was the only solution to remedy RCCL hangs at scale when using the OFI plugin, at the time of writing.

The code to run the model at inference is probablier easier to look at, if you just want to see the model architecture.
It can be found on all Huggingface repos of this model, and at `recpre/raven_modeling_minimal.py`.



## The grim details

What steps would you have to take if you were to replicate this model training run on an AMD cluster? Follow this outline (and please ask for more details when you're stuck).

1. Use `scripts/tokenizer_generation.py` to generate the tokenizer. Before you run the script, adapt all paths for your system. Data download is automatic. You also need the BPE trainer from https://github.com/gautierdag/bpeasy.
2. Run `scripts/scalable_data_download.py` to download all raw datasets. The name of the script is a lie, this is not so scalable, it will take a long time, lots of space and fail due to random errors. You'll also notice that there a number of extra rules hardcoded in the script for various badly formatting datasets. By the time you run this script, some of these authors may have updated their dataset breaking assumptions set here. You would get an error in that case, and would need to investigate that particular dataset. After this step, you'd have all raw datasets in a `staging` folder.
3. Run the `scripts/parquet_to_parquet_tokenizer.py` to generate the tokenized dataset. Again, remember to set your paths correctly.
4. After tokenizing, run the `scripts/parquet_to_parquet_shuffler.py` to shuffle the data.
5. Define your own launch config in `launch_configs/` or use our config, and launch `train.py` onto your cluster. We launched onto frontier with the launcher file called `launch_frontier.py`, but this will not help you on a different cluster. Follow your cluster's best practices and environment flag guidelines when setting up a large-scale run. The core command is just `python train.py --config=launch_configs/your_config.yaml`.
6. Watch it train (hopefully). You can add additional segments to the training run as needed.

We are currently looking to find a reliable way to host the entire pre-processed dataset, which would save you steps 1-4. We'll update here once this has happened.

## License
This code is released under an [apache-2.0](https://choosealicense.com/licenses/apache-2.0/)  license. Part of the code is licensed under the Lightning AI Apache-2.0 license.

## Citation
```
@article{geiping_scaling_2025,
  title = {Scaling up {{Test-Time Compute}} with {{Latent Reasoning}}: {{A Recurrent Depth Approach}}},
  shorttitle = {Scaling up {{Test-Time Compute}} with {{Latent Reasoning}}},
  author = {Geiping, Jonas and McLeish, Sean and Jain, Neel and Kirchenbauer, John and Singh, Siddharth and Bartoldson, Brian R. and Kailkhura, Bhavya and Bhatele, Abhinav and Goldstein, Tom},
  year = {2025},
  month = feb,
  eprint = {2502.05171},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2502.05171},
  url = {http://arxiv.org/abs/2502.05171},
  urldate = {2025-02-10},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Computation and Language,Computer Science - Machine Learning},
  journal = {arxiv:2502.05171[cs]}
}
```

## Contact
Please, feel free to contact us with any questions, or open an issue!
