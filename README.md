## Domain-level Pairwise Semantic Interaction for Aspect-Based Sentiment Classification (PAKDD 2022)

paper http://arxiv.org/abs/2202.10032



![image](https://user-images.githubusercontent.com/33895584/155280272-a2e177c1-c31a-4187-8bc0-bf6c8bcdb2b3.png)




**structure**

- <u>datasets</u>:  This directory stores restaurant14/Restaurant15/Restaurant16/Laptop15/Laptop16 processed datasets
- <u>models</u>: This directory stores our proposed PSI model
- data_utils_I_P.py: This python file is the sampling processing of datasets of Interacting Polarity
- train.py: This python file is the training function of PSI model 

To run train.py, you can input following command:
python train.py --dataset res14 --lr 2e-5 --n_classes 3 --n_samples 4

'--dataset' parameter can be set to res15/res16/laptop15/laptop16 datasets
'--lr' parameter means learning rate, it can be set to 5e-5, 2e-5
'--n_classes' parameter means the number of classes, it can be set to 2, 3 
'--n_samples' parameter means the number of samples per class, it can be set to 3, 4, 5 

\label{table2}
\begin{tabular}{c|cc|cc|cc|cc}
\hline
{\color[HTML]{333333} }                                                                       & \multicolumn{2}{c|}{{\color[HTML]{333333} Res14}}                                            & \multicolumn{2}{c|}{{\color[HTML]{333333} Lap15}}                           & \multicolumn{2}{c|}{{\color[HTML]{333333} Res16}}                             & \multicolumn{2}{c}{{\color[HTML]{333333} Lap16}}                              \\
\multirow{-2}{*}{{\color[HTML]{333333} Models}}                                           & {\color[HTML]{333333} Acc.}           & {\color[HTML]{333333} F1}              & {\color[HTML]{333333} Acc.}          & {\color[HTML]{333333} F1}            & {\color[HTML]{333333} Acc.}           & {\color[HTML]{333333} F1}             & {\color[HTML]{333333} Acc.}           & {\color[HTML]{333333} F1}             \\ \hline
{\color[HTML]{333333} TC-LSTM}                                                                & {\color[HTML]{333333} 0.781}          & {\color[HTML]{333333} 0.675}           & {\color[HTML]{333333} 0.745}         & {\color[HTML]{333333} 0.622}         & {\color[HTML]{333333} 0.813}          & {\color[HTML]{333333} 0.629}          & {\color[HTML]{333333} 0.766}          & {\color[HTML]{333333} 0.578}          \\
{\color[HTML]{333333} ATAE-LSTM}                                                              & {\color[HTML]{333333} 0.772}          & {\color[HTML]{333333} -}              & {\color[HTML]{333333} 0.747}         & {\color[HTML]{333333} 0.637}         & {\color[HTML]{333333} 0.821}          & {\color[HTML]{333333} 0.644}          & {\color[HTML]{333333} 0.781}          & {\color[HTML]{333333} 0.591}          \\
{\color[HTML]{333333} RAM}                                                                    & {\color[HTML]{333333} 0.802}          & {\color[HTML]{333333} 0.708}          & {\color[HTML]{333333} 0.759}         & {\color[HTML]{333333} 0.639}         & {\color[HTML]{333333} 0.839}          & {\color[HTML]{333333} 0.661}          & {\color[HTML]{333333} 0.802}          & {\color[HTML]{333333} 0.627}          \\
{\color[HTML]{333333} IAN}                                                                    & {\color[HTML]{333333} 0.793}          & {\color[HTML]{333333} 0.701}          & {\color[HTML]{333333} 0.753}         & {\color[HTML]{333333} 0.625}         & {\color[HTML]{333333} 0.836}          & {\color[HTML]{333333} 0.652}          & {\color[HTML]{333333} 0.794}          & {\color[HTML]{333333} 0.622}          \\
{\color[HTML]{333333} Clause-Level ATT}                                                       & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}               & {\color[HTML]{333333} 0.816}         & {\color[HTML]{333333} 0.667}         & {\color[HTML]{333333} 0.841}          & {\color[HTML]{333333} 0.667}          & {\color[HTML]{333333} 0.809}          & {\color[HTML]{333333} 0.634}          \\
{\color[HTML]{333333} \begin{tabular}[c]{@{}c@{}}LSTM+synATT\\ +TarRep\end{tabular}}          & {\color[HTML]{333333} 0.806}          & {\color[HTML]{333333} 0.713}           & {\color[HTML]{333333} 0.822}         & {\color[HTML]{333333} 0.649}         & {\color[HTML]{333333} 0.846}          & {\color[HTML]{333333} 0.675}          & {\color[HTML]{333333} 0.813}          & {\color[HTML]{333333} 0.628}          \\
{\color[HTML]{333333} kumaGCN}                                                                & {\color[HTML]{333333} 0.814}          & {\color[HTML]{333333} 0.736}           & {\color[HTML]{333333} -}             & {\color[HTML]{333333} -}             & {\color[HTML]{333333} 0.894}          & {\color[HTML]{333333} 0.732}          & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}              \\
{\color[HTML]{333333} RepWalk}                                                                & {\color[HTML]{333333} 0.838}          & {\color[HTML]{333333} 0.769}          & {\color[HTML]{333333} -}             & {\color[HTML]{333333} -}             & {\color[HTML]{333333} 0.896}          & {\color[HTML]{333333} 0.712}          & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}              \\
{\color[HTML]{333333} IMN}                                                                    & {\color[HTML]{333333} 0.839}          & {\color[HTML]{333333} 0.757}          & {\color[HTML]{333333} 0.831}         & {\color[HTML]{333333} 0.654}         & {\color[HTML]{333333} 0.892}          & {\color[HTML]{333333} 0.71}           & {\color[HTML]{333333} 0.802}          & {\color[HTML]{333333} 0.623}          \\ \hline
{\color[HTML]{333333} BERT}                                                                   & {\color[HTML]{333333} 0.867}          & {\color[HTML]{333333} 0.764}          & {\color[HTML]{333333} 0.818}         & {\color[HTML]{333333} 0.699}         & {\color[HTML]{333333} 0.884}          & {\color[HTML]{333333} 0.755}          & {\color[HTML]{333333} 0.817}          & {\color[HTML]{333333} 0.665}           \\
{\color[HTML]{333333} BERT-QA}                                                                & {\color[HTML]{333333} -}          & {\color[HTML]{333333} -}              & {\color[HTML]{333333} 0.827}         & {\color[HTML]{333333} 0.595}         & {\color[HTML]{333333} 0.896}          & {\color[HTML]{333333} 0.715}          & {\color[HTML]{333333} 0.812}          & {\color[HTML]{333333} 0.596}          \\
{\color[HTML]{333333} \begin{tabular}[c]{@{}c@{}}AC-MIMLLN\end{tabular}}              & {\color[HTML]{333333} 0.893}          & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}             & {\color[HTML]{333333} -}             & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}              & {\color[HTML]{333333} -}              \\ 
{\color[HTML]{333333} \begin{tabular}[c]{@{}c@{}}CoGAN\end{tabular}}              & {\color[HTML]{333333} -}          & {\color[HTML]{333333} -}              & {\color[HTML]{333333} 0.851}             & {\color[HTML]{333333} 0.745}             & {\color[HTML]{333333} \textbf{0.920}}              & {\color[HTML]{333333} \underline{0.816}}              & {\color[HTML]{333333} \underline{0.842}}              & {\color[HTML]{333333} 0.707}              \\\hline
{\color[HTML]{333333} \begin{tabular}[c]{@{}c@{}}PSI (BERT)\end{tabular}} & {\color[HTML]{333333} \underline{0.916}} & {\color[HTML]{333333}  \underline{0.857}}  & {\color[HTML]{333333} \underline{0.860}} & {\color[HTML]{333333} \underline{0.756}} & {\color[HTML]{333333} 0.901} & {\color[HTML]{333333} 0.788} & {\color[HTML]{333333} 0.839} & {\color[HTML]{333333} \underline{0.723}} \\
{\color[HTML]{333333} PSI (BERT-Large)}                                                               & {\color[HTML]{333333} \textbf{0.924}}          & {\color[HTML]{333333} \textbf{0.863}}           & {\color[HTML]{333333} \textbf{0.868}}         & {\color[HTML]{333333} \textbf{0.760}}         & {\color[HTML]{333333} \underline{0.913}}          & {\color[HTML]{333333} \textbf{0.828}}          & {\color[HTML]{333333} \textbf{0.87}}          & {\color[HTML]{333333} \textbf{0.737}}          \\
\hline
\end{tabular}
\end{table*}
