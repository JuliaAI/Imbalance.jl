---
title: 'Imbalance: A comprehensive, multi-interface, extensively documented Julia toolbox to address class imbalance'
tags:
  - machine learning
  - classification
  - class imbalance
  - resampling
  - oversampling
  - undersampling
  - julia

authors:
  - name: Essam Wisam
    orcid: 0009-0009-1198-7166
    equal-contrib: true
    affiliation: 1
  - name: Anthony Blaom
    orcid: 0000-0001-6689-886X
    equal-contrib: true 
    affiliation: 2

affiliations:
 - name: Cairo University, Egypt
   index: 1
 - name: University of Auckland, New Zealand
   index: 2

date: 11 October 2023
bibliography: paper.bib
---

# Summary

Given a set of observations that each belong to a certain class, supervised classification aims to learn a classification model that can predict the class of a new, unlabeled observation [@Cunningham:2008]. This modeling process finds extensive application in real-life scenarios, including but not limited to medical diagnostics, recommendation systems, credit scoring, and sentiment analysis.

In various real-world scenarios, such as those pertaining to the detection of particular conditions like fraud, faults, pollution, or rare diseases, a severe discrepancy between the number of observations in each class can occur. This is known as class imbalance. This poses a problem if assumptions inherent in the classification model imply hindered performance when the model is trained on imbalanced data as is commonly the case [@Ali:2015]. Two prevalent strategies for mitigating class imbalance, when it poses a problem to the classification model, involve either increasing the representation of less frequently occurring classes through oversampling or reducing instances of more frequently occurring classes through undersampling. It may be also possible to achieve even greater performance by combining both approaches [@Zeng:2016] or by resampling the data multiple times and training the classification model on each resampled dataset to form an ensemble model that aggregates results from different model instances [@Liu:2009].

## Imbalance.jl

In this work, we present, `Imbalance.jl`, a software toolbox implemented in the Julia programming language that offers over 10 well established techniques that help address the class imbalance issue. Additionally, we present a companion package, `MLJBalancing.jl`, which: (i)  facilitates the integration of resampling methods with classification models, to create a seamless machine learning pipeline that behaves like a single unified model;  and (ii) implements a general version of the EasyEnsemble algorithm presented in [@Liu:2009]. The set of resampling techniques implemented in `Imbalance.jl` and `MLJBalancing.jl` are shown \autoref{techniques}. Although no combination resampling techniques are explicitly presented, they are easy to form using the `BalancedModel` wrapper found in `MLJBalancing.jl`.


The toolbox offers a pure functional interface for each method implemented. For example, `SMOTE` can be used in the following fashion:

```julia
Xover, yover = smote(X, y)
```
Here `Xover, yover` are `X, y` after oversampling.

A `ratios` hyperparameter is almost always present to control the degree of oversampling or undersampling to be done for each class. All hyperparameters for a resampling method have default values that can be overridden. The following example shows how to use the `smote` method with a custom `ratios` hyperparameter:

```julia
X_after, y_after = smote(X, y; ratios=1.2)
```

The `ratios` hyperparameter controls the amount of oversampling or undersampling to be done for each class. When it is a float, each class will be oversampled or undersampled to the size of the majority or minority class respectively, multiplied by the float. Thus, `ratios=1.0` would oversample all classes to the size of the majority class or undersample all classes to the size of the minority class depending on the type of the `resample` technique. Alternatively,`ratios` can be a dictionary mapping each class label to the float ratio for that particular class.

: Resampling techniques implemented in `Imbalance.jl` and `MLJBalancing.jl`. []{label="techniques"}

| Technique                  | Type          | Supported Data Types          |
|----------------------------|---------------|-------------------------------|
| BalancedBaggingClassifier  | Ensemble      | Continuous and/or nominal         |
| Borderline SMOTE1          | Oversampling  | Continuous                    |
| Cluster Undersampler       | Undersampling  | Continuous                    |
| Edited Nearest Neighbors Undersampler | Undersampling  | Continuous |
| Random Oversampler         | Oversampling  | Continuous and/or nominal      |
| Random Undersampler        | Undersampling  | Continuous and/or nominal      |
| Random Walk Oversampler    | Oversampling  | Continuous and/or nominal      |
| ROSE                       | Oversampling  | Continuous                    |
| SMOTE                      | Oversampling  | Continuous                    |
| SMOTE-N                    | Oversampling  | Nominal                        |
| SMOTE-NC                   | Oversampling  | Continuous and nominal         |
| Tomek Links Undersampler   | Undersampling  | Continuous |

# Statement of Need

A substantial body of literature in the field of machine learning and statistics is devoted to addressing the class imbalance issue. This predicament has often been aptly labeled the "curse of class imbalance," as noted in [@Picek:2018] and [@Kubt:1997] which follows from the pervasive nature of the issue across diverse real-world applications and its pronounced severity; a classifier may incur an extraordinarily large performance penalty in response to training on imbalanced data.

The literature encompasses a myriad of oversampling and undersampling techniques to approach the class imbalance issue. These include SMOTE [@Chawla:2002] which operates by generating synthetic examples along the lines joining existing points, SMOTE-N and SMOTE-NC [@Chawla:2002] which are variants of SMOTE that can deal with categorical data. The sheer number of SMOTE variants makes them a body of literature on their own. Notably, the most widely cited variant of SMOTE is BorderlineSMOTE [@Han:2005]. Other well established oversampling techniques include RWO [@Zhang:2014] and ROSE [@Menardi:2012]. On the other hand, the literature also encompasses many undersampling techniques such as cluster undersampling [@Lin:2016] and condensed nearest neighbors [@Hart:1968]. Furthermore, methods that combine oversampling and undersampling [@Zeng:2016] or resampling with ensemble learning [@Liu:2009] are also present.

The existence of a toolbox with techniques that harness this wealth of research is necessary for the development of novel approaches to the class imbalance problem and for machine learning research in general. Aside from addressing class imbalance in a general machine learning research setting, the toolbox can help in class imbalance research settings by making it possible to juxtapose different methods, compose them together or form variants of them without having to reimplement them from scratch. In popular programming languages, such as Python, a variety of such toolboxes already exist, such as imbalanced-learn [@Lematre:2016] and SMOTE-variants [@Kovács:2019]. Meanwhile, Julia, a well known programming language with over 40M downloads [@DataCamp:2023], has been lacking a similar toolbox to address the class imbalance issue in general multi-class, heterogeneous data settings. This has served as the primary motivation for the creation of the `Imbalance.jl` toolbox.


# Imbalance.jl Design Principles

The toolbox implementation follows a specific set of design principles in terms of the implemented techniques, interface support, developer experience and testing, and user experience.

## Implemented Techniques 
- Should support all four major types of resampling approaches
- Should be generally compatible with multi-class settings
- Should offer solutions to heterogenous data settings (continuous and nominal data)
- When possible, preference should be given to techniques that are more common in the literature or industry

Methods implemented in the `Imbalance.jl` toolbox indeed meet all aforementioned design principles for the implemented techniques. The one-vs-rest scheme as proposed in [@Fernández:2013] was used to generalize the technique to multi-class when needed.

## Interface Support
- Should support both matrix and table inputs
- Target variable may or may not be given as a separate column
- Should expose a pure functional implementation, but also support popular Julia machine learning interfaces.
- Should be possible to wrap an arbitrary number of resampler models with an MLJ model to behave as a unified model using MLJBalancing

Methods implemented in the `Imbalance.jl` toolbox meet all the interface design principles above. It particularly implements the `MLJ` and `TableTransforms` interface for each method. `BalancedModel` from `MLJBalancing.jl` also allows fusing an arbitrary number of resampling models and a classifier together to behave as one unified model.


## Developer Experience and Testing

- Should document all functions, including internal ones
- Comments should be included to justify or simplify written implementations when needed
- Features commonly used by multiple resampling techniques should be implemented in a single function and reused
- There should exist a developer guide to encourage and guide contribution
- Functions should be implemented in smaller units to aid for testing
- Testing coverage should be maximized; even the most basic functions should be tested

This set of design principles is also satisfied by `Imbalance.jl`. Implemented techniques are tested by testing smaller units that form the technique. End-to-end tests are performed for each technique by testing properties and characteristics of the technique or by using the `imbalanced-learn` toolbox from Python and comparing outputs.

## User Experience

- Functional documentation should be comprehensive and clear
- Examples (with shown output) that work after copy-pasting should accompany each method
- An illustrative visual example that presents a plot or animation should preferably accompany each method
- A practical example that uses the method with real data should preferably accompany each method. Practical examples that study hyperparameter effects may also be provided.
- Users should be able to easily run the illustrative or practical examples (e.g., via Google Colab)
- If an implemented method lacks an online explanation, an article that explains the method after its implemented should be preferably written

The `Imbalance.jl` documentation indeed satisfies this set of design principles. Methods are associated with examples that can be copy-pasted, examples that demonstrate the operation of the technique visually and possibly, examples that use it with a real world dataset to improve the performance of a classification model.

## Author Contributions

Design: E. Wisam, A. Blaom. Implementation, tests and documentation: E. Wisam. Code and documentation review: A. Blaom.