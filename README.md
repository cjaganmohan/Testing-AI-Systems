# Testing-AI-Systems
My current research project -- testing autonomous vehicle systems

In this project, we propose a combinatorial approach to testing DNN models, specifically DNN models used in autonomous driving systems.

All experimental results are publicly available at: https://tinyurl.com/y2s6qxeo

Comparison experiments with deepTest - First, we measured the neuron coverage achieved by the seed image (baseline). Then, we measure the cumulative neuron coverage achieved by a test set generated
using deepTest approach. Finally, we measure the cumulative neuron coverage achieved using the t-way test .

### Rambo model: 
  * Cumulative neuron coverage deepTest approach - https://tinyurl.com/ybkpcsx9
  * Cumulative neuron coverage t-way test - https://tinyurl.com/3u8pz2x7

### Chauffeur model: 
  * Cumulative neuron coverage deepTest approach - https://tinyurl.com/r7ufsnf
  * Cumulative neuron coverage t-way test - https://tinyurl.com/tsudvvw8
  
### To perform a fair comparison, we select a subset of t-way tests such that number of t-way tests == number of tests generated using the deepTest approach

For each group, to reduce variations we generated five samples using different seeds. The seeds used in this selection process is available at: https://tinyurl.com/ku8d9ph4

Cumulative neuron coverage results achieved by subset of t-way tests:
  * Rambo model - https://tinyurl.com/9zzk5fe7
  * Chauffeur model - https://tinyurl.com/jz5je88w


For any queries, please email me at:  ***jaganmohan[dot]chandrasekaran[at]mavs[dot]uta[dot]edu***
