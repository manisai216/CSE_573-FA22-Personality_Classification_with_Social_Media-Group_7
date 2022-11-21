# CSE_573-FA22-Personality_Classification_with_Social_Media-Group_7
CSE 573 - Fall 2022 SWM Project

Group Members:

1. Bhavishya Puttagunta
2. Bindu Trilekha Mylavarapu
3. Dhana Satya Aneesh Ravipati
4. Haripriyanga Navaneethan
5. Mani Sai Tejaswy Valluri
6. Sai Cherish Potluri

Dataset Link - https://drive.google.com/file/d/15v2umVbv1OarPfNrePH9V-KPk7qqA6Cg

Reference Paper - https://psycnet.apa.org/fulltext/2016-57141-003.pdf

Steps to Run:

1. Clone Repository to local system

2. Download three files (users.csv, users-likes.csv, likes.csv) from the given dataset link above and place the three files in the repo's Data folder. 

2. Download and install R in your system

3. In your terminal, go to the Code folder and run the following code files:

   a. Run command - "Rscript SVD_Cluster_Analysis.R" for SVD dimensionality reduction and visualizing K-Means and DBSCAN algorithms for SVD clusters.
   
   b. Run command - "Rscript LDA_Cluster_Analysis.R" for LDA dimensionality reduction and visualizing K-Means and DBSCAN algorithms for LDA clusters.
   
   c. Run command - "Rscript Prediction_Analysis.R" for performing Linear and Logistic Regression algorithms to predict user personality types.
   
4. Evaluations folder contains the resulting R visualization files from running each of the above files in the code folder.
