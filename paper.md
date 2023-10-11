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
  - name: Essam W. Amin
    orcid: 0009-0009-1198-7166
    equal-contrib: true
    affiliation: 1
  - name: Anthony Blaom
    orcid: 0000-0000-0000-0000
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

Given a set of observations that each belong to certain class, supervised classification aims to learn a classification model that can predict the class of a new, unlabeled observation [@Cunningham:2008]. This modeling process finds extensive application in real-life scenarios, including but not limited to medical diagnostics, recommendation systems, credit scoring, and sentiment analysis.

In various real-world scenarios, such as those pertaining to the detection of particular conditions like fraud, faults, pollution, or rare diseases, a severe discrepancy between the number of observations in each class can occur. This is known as class imbalance. Assumptions inherent in the classification model may imply hindered performance for the trainedm odel when class imbalance is present in the training data [@Ali:2015]. Two prevalent strategies for mitigating class imbalance, when it poses a problem to the classification model, involve either increasing the representation of data in the smaller classes through oversampling or reducing the data in the larger classes through undersampling. It may be also possible to achieve even greater performance by combining both approaches [@Zeng:2016] or by resampling the data multiple times and training the classification model on each resampled dataset to form an ensemble model that aggregates results from different model instances [@Liu:2009].

## Imbalance.jl

In this work, we present, `Imbalance.jl`, a software toolbox implemented in the Julia programming language that offers over 10 well established techniques that help address the class imbalance issue. Additionally, we present a companion package, `MLJBalancing.jl`, which facilitates the integration of resampling methods with classification models, to create a seamless machine learning pipeline that behaves like a single unified model and implements a general version of the EasyEnsemble algorithm presented in [@Zeng:2016]. The set of resampling techniques implemented in `Imbalance.jl` and `MLJBalancing.jl` are shown \autoref{techniques}. Although no combination resampling techniques are explicitly presented, they are easy to form using the `BalancedModel` wrapper found in `MLJBalancing.jl`.

: Resampling techniques implemented in Imbalance.jl and MLJBalancing.jl. []{label=”techniques”}

| Technique                  | Type          | Basic Mechanism                      | Supported Data Types          |
|----------------------------|---------------|--------------------------------------|-------------------------------|
| BalancedBaggingClassifier  | Ensemble      | Delete existing data to form different subsets    | Continuous and/or nominal         |
| Borderline SMOTE1          | Oversampling  | Generate synthetic data               | Continuous                    |
| Cluster Undersampler       | Undersampling  | Generate new data or delete existing data  | Continuous                    |
| Edited Nearest Neighbors Undersampler | Undersampling  | Delete existing data meeting certain conditions (cleaning) | Continuous |
| Random Oversampler         | Oversampling  | Repeat existing data                  | Continuous and/or nominal      |
| Random Undersampler        | Undersampling  | Delete existing data         | Continuous and/or nominal      |
| Random Walk Oversampler    | Oversampling  | Generate synthetic data               | Continuous and/or nominal      |
| ROSE                       | Oversampling  | Generate synthetic data               | Continuous                    |
| SMOTE                      | Oversampling  | Generate synthetic data               | Continuous                    |
| SMOTE-N                    | Oversampling  | Generate synthetic data               | Nominal                        |
| SMOTE-NC                   | Oversampling  | Generate synthetic data               | Continuous and nominal         |
| Tomek Links Undersampler   | Undersampling  | Delete existing data meeting certain conditions (cleaning) | Continuous |

The package offers a pure functional interface for each method implemented. An arbitrary `Imbalance.jl` resampling method `resample` can be used in the following fashion:

```julia
X_after, y_after = resample(X, y)
```

A `ratios` hyperparameter is almost always present to control the degree of oversampling or undersampling to be done for each class. All hyperparameters for a resampling method have default values. To avoid using the defaults, an arbitrary `Imbalance.jl` resampling method `resample` that also has other hyperparameters `a`, `b` and `c` can be used in the following fashion:

```julia
a = ...
ratios = 1.0
b = ...
c = ...
X_after, y_after = resample(X, y; a, ratios, b, c)
```

`ratios` is a hyperparameter that controls the amount of oversampling or undersampling to be done for each class, when it is a float, each class will be oversampled or undersampled to the size of the majority or minority class respectively times the float. Thus, `ratios=1.0` would oversample all classes to the size of the majority class or undersample all classes to the size of the minority class depending on the type of the `resample` technique. `ratios` can also be a dictionary mapping each class label to the float ratio for that particular class.


# Statement of Need

A substantial body of literature in the field of machine learning and statistics is devoted to addressing the class imbalance issue. This predicament has often been aptly labeled the "curse of class imbalance," as noted in [Picek:2018] and [Kubt:1997] which follows from the pervasive nature of the issue across diverse real-world applications and its pronounced severity; a classifier may incur an extraordinarily large performance penalty in response to training on imbalanced data.

The literature encompasses a myriad of oversampling and undersampling techniques to approach the class imbalance issue. These include SMOTE [@Chawla:2002] which operates by generating synthetic examples along the lines joining existing points, SMOTE-N and SMOTE-NC [@Chawla:2002] which are variants of SMOTE that can deal with categorical data. The sheer number of SMOTE variants makes them a body of literature on their own. Notably, the most widely cited variant of SMOTE is BorderlineSMOTE [@Han:2005]. Other well established oversampling techniques include RWO [@Zhang:2014] and ROSE [@Menardi:2012]. On the other hand, the literature also encompasses many undersampling techniques such as cluster undersampling [@Lin:2016] and condensed nearest neighbors [@Hart:1968]. Furthermore, methods that combine oversampling and undersampling [@Zeng:2016] or resampling with ensemble learning [@Liu:2009] are also present.

The existence of a toolbox with techniques that harness this wealth of research is necessary for the development of novel approaches to the class imbalance problem and for machine learning research in general. Aside from addressing class imbalance in a general machine learning research setting, the toolbox can help in class imbalance research settings by making it possible to juxtapose different methods, compose them together or form variants of them without having to reimplement them from scratch. In popular programming languages, such as Python, a variety of such toolboxes already exist, such as imbalanced-learn [@Lematre:2016] and SMOTE-variants [@Kovács:2019]. Meanwhile, Julia, a well known programming language with over 40M downloads [@DataCamp:2023], has been lacking a similar toolbox to address the class imbalance issue in general multi-class, heterogeneous data settings. This has served as the primary motivation for the creation of the `Imbalance.jl` toolbox.


# Imbalance.jl Design Principles

The toolbox implementation follows a specific set of design principles in terms of the implemented techniques, interface support, developer experience and testing, and user experience.

## Implemented Techniques 
- Should support all four major types of resampling approaches
- Should be generally compatible with multi-class settings
- Should offer solutions to heterogenous data settings (continuous and nominal data)
- When possible, preference should be given to techniques that are more common in the literature or industry

Methods implemented in the `Imbalance.jl` toolbox indeed meet all aforementioned design principles for the implemented techniques. The one-vs-rest scheme as proposed in [@Fernández:2013] which was used to generalize the technique to multi-class when needed.

## Interface Support
- Should support both matrix and table inputs
- Target variable may or may not be given separately
- Should support a pure functional interface and other interfaces common in Julia packages such as `MLJ`, `FeatureTransforms` and `TableTransforms`
- Should be possible to wrap an arbitrary number of resampler models with an MLJ model to behave as a unified model using MLJBalancing

Methods implemented in the `Imbalance.jl` toolbox meet all the interface design principles above. It particularly implements the `MLJ` and `TableTransforms` interface for each method. An arbitrary resampling method `resample` with a pure functional interface that is operated in this way:

```julia
X_after, y_after = resample(X, y; a, ratios,  b, c)
```

is equivalent to an `MLJ` interface that is operated in this way. 

```julia
Resampler = @load Resampler pkg=Imbalance

resampler = Resampler(a, ratios,  b, c)

mach = machine(oversampler)

Xover, yover = transform(mach, X, y)
```

where `Resampler` is the equivalent model name specified for `resample`. This is further equivalent to a `TableTransforms` interface that is operated in this way:

```julia
resampler = Resampler(y_ind; a, ratios,  b, c)
Xyover = Xy |> resampler                 
Xyover, cache = TableTransforms.apply(resampler, Xy)    
```
The `TableTransforms` interface does not assume that the target `y` is given separately. It rather assumes that it is oneof the columns given in `Xy`, specifically the column given by `y_ind`.

It is also possible to wrap an arbitrary number of resamplers with a machine learning model using the `BalancedModel` construct from `MLJBalancing` by simply passing the model, followed by the resamplers, as keyword arguments.
```julia
balanced_model = BalancedModel(model=model; balancer1=resampler1, balancer2=resampler2, ...)
```
In this, during training, data is passed to balancer1 and the result is passed to balancer2 and so on. The final result from the resampling pipeline is then passed to the model for training. During prediction, the balancers have no effect.

## Developer Experience and Testing

- Should document all functions, including internal ones
- Comments should be included to justify or simplify written implementations when needed
- Features commonly used by multiple resampling techniques should be implemented in a single function and reused
- There should exist a developer guide to encourage and guide contribution
- Functions should be implemented in smaller units to aid for testing
- Testing coverage should be maximized; even the most basic functions should be tested

This set of design principles is also satisfied by `Imbalance.jl`. Redundancy is generally avoided and common functionality such as generalizing to multiple classes or dealing with table inputs are centralized. Implemented techniques are tested by testing smaller units that form the technique. End-to-end tests are performed for each technique by testing properties and characteristics of the technique or by using the `imbalanced-learn` package from Python and comparing outputs.

## User Experience

- Functional documentation should be comprehensive and clear
- Examples (with shown output) that work after copy-pasting should accompany each method
- An illustrative visual example that presents a plot or animation should preferably accompany each method
- A practical example that uses the method with real data should preferably accompany each method. Practical examples that study hyperparameter effects may also be provided.
- Users should be able to easily run the illustrative or practical examples (e.g., via Google colab)
- If an implemented method lacks an online explanation, an article that explains the method after its implemented should be preferably written

All techniques initially implemented in `Imbalance.jl` come with an example with shown outputs that can be copy-paste and run for the three interfaces, and almost always, also another example that shows a visual plot for the output from the algorithm on synthetic data  and an animation for the algorithm operation. There are also initially a set of 9 tutorials that use `Imbalance.jl` techniques with real world dataset to improve model performance or study hyperparameter effects. All techniques initially implemented in `Imbalance.jl` also correspond to an article on the Medium website written by the author that explains how the technique works.