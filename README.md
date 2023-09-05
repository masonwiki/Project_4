# Project_4

Members: Steven U, Mason W, Ted B

The chatbot requires a .pth file to work. This file could not have been included into the repo since it was too big. Running "train.py" will create your own data.pth which can be used to test our chatbot.

## Powerpoint link: (pngs of slides are located in Project-4 Presentations folder along with Website visuals)

## Code Info / Data Process
Raw data was retrieved in JSON format. Data cleanup was preformed within Python using SQlite. All data was stored and sorted in SQlite database.
Machine learning was done through the use of Pytorch and done within multiple python scripts for ease of readiablity

## Libraries Used
- SQLite (Data cleanup and storage)
- Pytorch (ML/NLP using their integrated Nvidia CUDA GPU Processing)
- Scikit-Learn (Training and Testing data)
- Flask (Website Development)

## Summary
Although we did not get the desired results wanted for our project, we did make good headway. After data cleanup and preprossessed JSON we trained our model with Pytorch. Each training session took roughly 3 hours due to the large dataset. We took advantage of Pytorch's CUDA function which uses Nvida GPU's to train data faster than using the computers CPU. We were able to train with 0.000 loss. Although training gave us the desired results, the chatbot did not. The bot responds incoherently and with little relation to the user's prompt.
