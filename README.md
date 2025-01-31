# DINAMO: Dynamic and INterpretable Anomaly MOnitoring for Large-Scale Particle Physics Experiments



## Getting started

To get started with a limited amount of datasets, run the following command:

```bash
python3 scripts/run_all.py --n_datasets 10 --parallelized 1 --n_cpus 5
```

This command will perform the following tasks:
- Install the package
- Create all needed directories (to store data, plots, and results)
- Generate toy datasets and save them in data/
- Produce plots to describe the datasets (class ratio, runs statistics distributions, uncertainty visualization, time evolution of the parameters of the Gaussians, dead bins distributions) and save them in plots/data_eda/.
- Run both standard and ML approaches on the datasets
- Validate the results and save the corresponding plots and outputs in plots/\*_approach/ and results/\*_approach/, respectively

All parameters are taken from the configuration files.

To produce the aggregated results over all the analysed datasets, run the following command
```bash
python3 scripts/run_aggregate_results.py
```

The plots showing the aggregated results will be saved in plots/ and the corresponding tables with all the numbers in results/*/metrics

## License

This project is licensed under the [MIT](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the members of the LHCb Collaboration for useful discussions that helped improve this work. We are also grateful to CloudVeneto for providing IT support and GPU resources. Arsenii Gavrikov is supported by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101034319 and from the European Union – NextGenerationEU. Julián García Pardiñas is supported by the U.S. National Science Foundation under Grant No. 2411204. This work has also been supported by the National Resilience and Recovery Plan (PNRR) through the National Center for HPC, Big Data and Quantum Computing.
