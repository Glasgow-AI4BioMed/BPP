# BPP
BPP: A Platform for Automatic Biochemical Pathway Prediction
> Authors: Xinhao Yi, Siwei Liu, Yu Wu, Douglas McCloskey*, Zaiqiao Meng*

## Contributions
1. We develop BPP, an open-source biochemical pathway analysis platform dedicated to predicting potential links and node attributes in biochemical pathway networks.

2.  Based on BPP, we evaluate the performance of four representation learning models on four biochemical pathway datasets. Experimental results suggest that these automated prediction models can achieve reliable performance on link prediction and attribute prediction task.

3. BPP integrates an explainer that provides an interpretation of the prediction results, i.g., offering the contribution of nodes and attributes within the reaction for current prediction result.

4. We verify the effectiveness of BPP by conducting a case study on SARS-CoV-2's invasion process. The results indicate that BPP can successfully identify unseen links within pathways.

## Motivation

1. Identifying potential links in biochemical pathway networks is essential for targeting disease markers, discovering drug targets, reconfiguring metabolic networks and addressing gaps in pathways holes in biosynthesis.

2. Traditional experimental methods can impose significant time and labour burdens on researchers, due to a vast number of candidates, consequently, our goal is to enhance the efficiency of pathway studies.

## Website
[Link to BPP Platform](http://52.146.12.46:5000)

## User Guidance
In this part, we'll introduce how to use our plaform. 

### Customise Biochemical Reaction
The key feature in BPP is to customise a biochemical reaction.

![customise the reaction](./figures/cus_reaction.png "customise the reaction")

Then, you freely select the dataset, customise biochemical reaction and then make prediction. 

![customise the reaction](./figures/cus_reaction_detail.png "customise the reaction detail")

Currently, you can choose which of the four datasets - Disease, Metabolism, immune system and signal transduction - you want to study, and then, feel free to select and load the biochemical reactions in that dataset. the biochemical reaction will be presented as a dynamic graph where blue nodes represent input entities, red nodes represent output entities and yellow triangles represent that biochemical reaction itself.

![customise the reaction](./figures/cus_load_reaction.png "load the reaction")

Or you can choose not to load an existing biochemical reaction and instead just create a new blank biochemical reaction by clicking New Reaction. Next, you have the option to freely add biochemical entities. You can select the ones you are interested in from the thousands of biochemical entities in this dataset and add them as inputs or inputs to the current biochemical reaction. This process allows you to narrow down the biochemical entities using the search box we provide.

![customise the reaction](./figures/cus_load_node.png "load node")








