#import "template.typ": *

#let in-outline = state("in-outline", false)
#let flex-caption(long, short) = context if in-outline.get() { short } else { long }
#show outline: it => {
  in-outline.update(true)
  it
  in-outline.update(false)
}

#let labeled(equation, label) = {
  math.equation(
    block: true, 
    numbering: (x) => label, 
    equation
  )
}

#show: project.with(
  title: [Clustering],
  header_title: "Clustering",
  subject: "Optimization in Data Science",
  authors: ("Víctor Diví i Cuesta", "Ivan Parreño Benítez"),
  outlines: (
    (
      title: "List of Figures",
      target: image
    ),
    (
      title: "List of Tables",
      target: table
    ),
    (
      title: "List of Codes",
      target: "Code"
    ),
  ),
  use-codly: false
)


= Introduction

In this project, several clustering methods are explored and compared using the Palmer Penguins Dataset (as both a multi and bivariate dataset). The project is structured as follows: @section:dataset presents and explains the used dataset, @section:clustering presents the methods used and their results, and @section:comparison shows a comparison of the results obtained with every method.

= Dataset <section:dataset>

The Palmer Penguins dataset #cite(<penguins>) is an introductory dataset widely used in data science as an alternative to the well-known "Iris" dataset.
The dataset is composed of data about three penguin species observed in the Palmer Archipelago, Antarctica.


In this project, we use a curated dataset that is provided in the `palmerpenguins` R#footnote[https://allisonhorst.github.io/palmerpenguins/articles/intro.html] and Python#footnote[https://github.com/mcnakhaee/palmerpenguins] packages.
In particular, it includes morphological and demographic measurements of 344 individual penguins, collected between 2007 and 2009 in the Palmer Long-Term Ecological Research #footnote[https://pallter.marine.rutgers.edu/], from three different species:

#grid(
  columns: (auto, 1fr),
  align: horizon,
  [
    - Adélie (_Pygoscelis adeliae_)
    - Chinstrap (_Pygoscelis antarcticus_)
    - Gentoo (_Pygoscelis papua_)
  ],
  grid.cell(
    align: center+horizon,
    {
      show figure.caption: set text(size: 0.8em)
      figure(
        image("images/penguins.png", width: 7cm),
        caption: [The Palmer Archipelago penguins. Artwork by \@allison_horst.]
      )
    }
  )
)

The curated dataset contains the following features (some names differ from the raw data):

#figure(
  table(
    columns: 3,
    row-gutter: (3pt, 3pt, auto),
    [*Feature*], [*Description*], [*Units/Values*],
    [`species`], [Penguin species], [`Adelie, Chinstrap, Gentoo`],
    [`island`], [Island in the Palmer Archipelago], [`Biscoe, Dream, Torgersen`],
    [`bill_length_mm`], [Length of the penguin's bill], [millimeters],
    [`bill_depth_mm`], [Depth of the penguin's bill], [millimeters],
    [`flipper_length_mm`], [Length of the penguin's flipper], [millimeters],
    [`body_mass_g`], [Penguin body mass], [grams],
    [`sex`], [Biological sex of the penguin], [`male, female`],
    [`year`], [Year of the observation], [`2007, 2008, 2009`]
  ),
  caption: [Features of the Palmer Penguins Dataset]
) <table:dataset>


Some preprocessing has been performed on the dataset before applying the clustering algorithms:

- *Data cleaning:* 11 measurements that contained null values in at least one feature have been removed.
- *Categorical encoding*: categorical features (`island`, `sex`, and `year`) have been transformed into numerical ones using a dummy encoding, increasing the number of features from 8 to 10.
- *Normalization*: all features have been normalized applying a Standard Score (or Z-Score) Normalization, since it is not affected by outliers in the dataset.

To decide which features to use in both the multivariate and the bivariate clustering, we have analyzed the correlation between all the features and  performed an Analysis of Variance (ANOVA) to analyze the degree of linear dependency between the different features and the species.

@figure:correlation shows the correlation matrix as a heatmap, where colors close to white indicate no correlation, while dark red and blue indicate strong correlation (positive and negative respectively). As can be seen, many of the features are significantly correlated (e.g. `flipper_length_mm` with `bill_depth_mm` and `body_mass`), and some features are also heavily correlated with the species (e.g. `bill_length_mm` with the Adélie species and `flipper_length_mm` with both Adélie and Gentoo species). We can also see that some features have almost no correlation with the species, namely the sex of the penguins and the year of the measurement, although they do have some correlation with the other features. These insights are further supported by the ANOVA results shown in @table:anova, which presents the F-Statistic for each feature and the related P-Value (higher F-Statistic indicate significant differences of the feature among species and the P-Value indicate the probability of this differences being by chance).

#grid(
  columns: 2,
  [
    #figure(
      image("images/correlation.png"),
      caption: [Correlation matrix of dataset features.]
    ) <figure:correlation>
  ],
  [
    #show table.cell: set text(size: 0.8em)
    #set table(align: (x, y) => if y == 0 {center + horizon} else { right })
    
    #figure(
      table(
        columns: 3,
        row-gutter: (3pt, auto),
        [*Feature*], [*F-Statistic*], [*P-Value*],
        [`flipper_length_mm`], [`567.406992`], [`1.587418e-107`],
        [`bill_length_mm`], [`397.299437`], [`1.380984e-088`],
        [`bill_depth_mm`], [`344.825082`], [`1.446616e-081`],
        [`body_mass_g`], [`341.894895`], [`3.744505e-081`],
        [`island_Dream`], [`208.347193`], [`3.063542e-059`],
        [`island_Torgersen`], [`43.988989`], [`1.160083e-017`],
        [`year_2008`], [`1.245831`], [`2.890514e-001`],
        [`sex_male`], [`0.024088`], [`9.762014e-001`],
        [`year_2009`], [`0.019740`], [`9.804544e-001`],
      ),
      caption: [ANOVA F-Values and related P-Values of each feature.]
    );
    <table:anova>
  ]
)

For the bivariate dataset, we have selected `flipper_length_mm` and `bill_length_mm`, which are the most significant features according to the ANOVA.

Due to the low relevance of the `sex` and `year` features, they have been discarded from the multivariate dataset.
Although some extra treatment of the dataset could be perform, such as a Principal Component Analysis (PCA) to reduce dimensionality and minimize correlation between features, we decided against it in favor of having features with real-life meaning.

#pagebreak()

= Clustering <section:clustering>

In this section, multiple clustering methods are explored, both exact and heuristic.
For each method, three different clusterings have been performed: MultiVariate, MultiVariate Scaled, and BiVariate.

The MultiVariate and BiVariate scenarios use the normalized dataset with the 6 and 2 selected features respectively. However, since we are using an euclidean distance for all the methods, we are giving the same weight to every dimension, even though we showed in the previous section that some features are significantly more relevant than others. Taking this into account, and to explore how a different distance affects the clusterings, we introduce the MultiVariate Scaled scenario, which adds "weights" to the most important features so they become more relevant in the clustering. To do so, we simply scale the features according to their weight, so that the resulting euclidean distance is also multiplied. The weights used are shown in @table:weights.

#figure(
  table(
    columns: 2,
    row-gutter: (3pt, auto),
    [*Feature*], [*Multiplying Factor*],
    [`flipper_length_mm`], [3],
    [`bill_length_mm`], [2],
    [`bill_depth_mm`], [1.75],
    [`body_mass_g`], [1.75],
  ),
  caption: flex-caption[Feature weights for the MultiVariate Scaled scenario. Other features have weight of 1.][Feature weights for the MultiVariate Scaled scenario.]
) <table:weights>

All functions using the different clustering methods are provided in independent files named accordingly to the method they use.

== Exact K-Medoids <section:kmedoids-ampl>

#figure(
  [
    #line(length: 90%)
    #raw(read("code/ampl.mod"), lang: "ampl", block: true)
    #line(length: 90%)
  ],
  caption: [AMPL Model for the K-Medoids.],
  supplement: "Code",
  kind: "Code",
) <code:extensive>
The K-Medoids formulation problem was implemented in AMPL, using the formulation presented in class, where every point has a distance to every other point in the clustering.  

To represent the assignments, a matrix of size n × n is used, where each entry indicates whether a point is assigned to a specific medoid.  

The constraints are as follows:

#list(
  [*OneAssignment:* Each point must be assigned to one and only one medoid.],
  [*KClusters:* Only k points can be medoids. In the x matrix, this means that only k diagonal elements (where i = j) take the value 1.],
  [*ClusterExists:* A point can only be assigned to a cluster if that cluster (medoid) actually exists.]
)

Finally, the objective function minimizes the total distance between each point and its assigned medoid.

#figure(
  image("images/ampl_clustering.png"),
  caption: flex-caption[K-Medoids using AMPL clustering results comparing from left to right, real clusters, MultiVariate results, MultiVariate Scaled results and BiVariate results.][K-Medoids AMPL clustering results]
) <clusters:kmedoids-ampl>


#figure(
  image("images/ampl_confusion.png"),
  caption: flex-caption[K-Medoids using AMPL clustering confusion matrices of MultiVariate results (left), MultiVariate Scaled results (middle), and BiVariate results (right).][K-Medoids AMPL confusion matrices]
) <confusion:kmedoids-ampl>



== Minimum Spanning Tree <section:mst>

The function for generating a Minimum Spanning Tree (MST) is shown in @code:mst, and uses the `mst_clustering` library, which automatically builds the MST from the data #cite(<VanderPlas2016>). This library allows us to adjust several parameters to control the clustering process:

- *cutoff*: determines how many edges in the MST are removed. This results in the creation of cutoff + 1 clusters.

- *metric*: specifies whether the input data already represents a distance matrix or whether the library should compute the distances itself.

- *approximate*: if set to TRUE , the algorithm computes an approximate MST instead of an exact one. This can speed up the process for large datasets but may slightly affect precision, so we decided to compute the best possible MST.

#figure(  
  [
    #line(length: 90%)
    #raw(read("code/mst.py"), lang: "python", block: true)
    #line(length: 90%)
  ],
  supplement: "Code",
  kind: "Code",
  caption: [MST Clustering using the `mst_clustering` library.]
) <code:mst>


@clusters:mst_clustering_library shows an example in which it is quite easy to see which edges will be cut if we create an MST, given that the 4 clusters are well isolated.

/* poner imagen de la libreria de de mst y comparar con la nuestra*/
#figure(
image("images/mst_clustering_library.png", width: 70%),
  caption: flex-caption[Example of a clustering with four groups where using k+1 as cut yields the same number of clusters.][Clustering library example]
) <clusters:mst_clustering_library>

As shown in the MST plot (@clusters:mst_clustering), the results are not very promising. In the bivariate case, the method generates three clusters—two containing a single point and one large cluster with the rest. For the scaled data, it detects only one point from the Chinstrap group and forms two large clusters with the remaining points. In the multivariate case, the method struggles with the Adelie group, failing to separate it properly. This occurs because outliers create large distance gaps, leading to poor cluster separation overall.


#figure(
image("images/mst_clustering.png"),
   caption: flex-caption[MST clustering results comparing, from left to right, real clusters, MultiVariate results, MultiVariate Scaled results and BiVariate results.][MST clustering results]
) <clusters:mst_clustering>

#figure(
image("images/mst_confusion.png"),
  caption: flex-caption[MST clustering confusion matrices of MultiVariate results (left), MultiVariate Scaled results (middle), and BiVariate results (right).][MST confusion matrices]
) <confusion:mst_clustering>

In order to obtain better results, we implemented a heuristic that treats clusters with only one point as outliers. We keep applying this process until we obtain cutoff + 1 clusters, each with a minimum size of 10. However, as can be observed, the results are still not very promising. The minimum size threshold helps to merge small but valid clusters, but the MST structure is sensitive to local density variations. This heurisitic also considers that points that do not belong to any cluster (-1 in the plots), are considered outliers and noise.

#figure(
image("images/msth_clustering.png"),
   caption: flex-caption[MST clustering heuristic results comparing, from left to right, real clusters, MultiVariate results, MultiVariate Scaled results and BiVariate results.][MST-H clustering results]
) <clusters:msth_clustering>

#figure(
image("images/msth_confusion.png"),
  caption: flex-caption[MST clustering heuristic confusion matrices of MultiVariate results (left), MultiVariate Scaled results (middle), and BiVariate results (right).][MST-H confusion matrices]
) <confusion:msth_clustering>


== K-Means <section:kmeans>

The K-Means clustering has been performed using the `scikit-learn` library using the default initialization, the Greedy K-Means++ #cite(<kmeanspp>), which samples the dataset using the probability distribution of the points’ contribution to the overall inertia#footnote[Extracted from scikit-learn's documentation: #link("https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html")].

// @code:kmeans shows the function that clusters a given dataset into `k` clusters.

// #figure(  
//   [
//     #line(length: 90%)
//     #raw(read("code/kmeans.py"), lang: "python", block: true)
//     #line(length: 90%)
//   ],
//   supplement: "Code",
//   kind: "Code",
//   caption: [K-Means Clustering using the `scikit-learn` library.]
// ) <code:kmeans>

@clusters:kmeans shows the results of the K-Means clustering for the three scenarios, while @confusion:kmeans shows their confusion matrices.
As can be clearly seen, the MultiVariate scenario groups all Gentoo penguins perfectly, with neither false positives nor false negatives, and classifies all Chinstrap correctly, but misclassifies more than one third of the Adélie penguins as Chinstrap.

Nothing of this happens in the Scaled or the BiVariate scenarios, in which every cluster contains the majority of the penguins of the corresponding species, but certain individuals from all three species are misclassified.

#figure(
image("images/kmeans_clustering.png"),
caption: flex-caption[K-Means clustering results comparing, from left to right, real clusters, MultiVariate results, MultiVariate Scaled results and BiVariate results.][K-Means clustering results]
) <clusters:kmeans>


#figure(
  image("images/kmeans_confusion.png"),
  caption: flex-caption[K-Means clustering confusion matrices of MultiVariate results (left), MultiVariate Scaled results (middle), and BiVariate results (right).][K-Means confusion matrices]
) <confusion:kmeans>


== K-Medoids <section:kmedoids>

The K-Medoids clustering has been computed using the `kmedoids` Python library using the default configuration, which uses the FasterPAM algorithm #cite(<fasterpam>).
// @code:kmedoids shows the function that clusters a given dataset into `k` clusters using the euclidean distances.

// #figure(  
//   [
//     #line(length: 90%)
//     #raw(read("code/kmedoids.py"), lang: "python", block: true)
//     #line(length: 90%)
//   ],
//   supplement: "Code",
//   kind: "Code",
//   caption: [K-Medoids Clustering using the `kmedoids` library.]
// ) <code:kmedoids>

@clusters:kmedoids shows the results of the K-Means clustering for the three scenarios, while @confusion:kmedoids shows their confusion matrices. The obtained results are almost identical to the ones obtained with the K-Means clustering in @section:kmeans. In the MultiVariate scenario, the results are exactly the same, while in the two others the results are slightly worse (7 and 2 more misclassified penguins respectively).


#figure(
image("images/kmedoids_clustering.png"),
  caption: flex-caption[K-Medoids clustering results comparing, from left to right, real clusters, MultiVariate results, MultiVariate Scaled results and BiVariate results.][K-Medoids clustering results]
) <clusters:kmedoids>


#figure(
  image("images/kmedoids_confusion.png"),
  caption: flex-caption[K-Medoids clustering confusion matrices of MultiVariate results (left), MultiVariate Scaled results (middle), and BiVariate results (right).][K-Medoids confusion matrices]
) <confusion:kmedoids>

#highlight[]
= Comparison <section:comparison>
In this section, we present a comparison of the speed and results of the different clustering methods. A summary of the results across all clustering methods and scenarios segregated by species is shown in @table:results. The total amount of correctly classified and misclassified penguins is also shown.

First, it is evident that the heuristic MST method performs poorly in terms of clustering quality. On the other hand, the heuristic versions of K-Means and K-Medoids give comparable results, which shows that their approximations are reasonable. The exact K-Medoids algorithm produces almost the same results as its heuristic counterpart, so it makes sense to look at computational times to decide which approach is better.

#let data = (
  "K-Medoids Exact": (
    "MV": (47, 99, 0, 0, 68, 0, 0, 0, 119),
    "MVS": (118, 27, 1, 6, 62, 0, 0, 0, 119),
    "BV": (140, 4, 2, 5, 58, 5, 0, 1, 118),
  ),
  "K-Means": (
    "MV": (47, 99, 0, 0, 68, 0, 0, 0, 119),
    "MVS": (131, 15, 0, 6, 62, 0, 0, 0, 119),
    "BV": (141, 4, 1, 5, 59, 4, 0, 1, 118),
  ),
  "K-Medoids": (
    "MV": (47, 99, 0, 0, 68, 0, 0, 0, 119),
    "MVS": (118, 27, 1, 6, 62, 0, 0, 0, 119),
    "BV": (140, 4, 2, 5, 58, 5, 0, 1, 118),
  ),
  "MST": (
    "MV": (47, 55, 44, 0, 68, 0, 0, 0, 119),
    "MVS": (146, 0, 0, 67, 1, 0, 0, 0, 119),
    "BV": (146, 0, 0, 67, 1, 0, 118, 0, 1),
  ),
  "MST-H": (
    "MV": (47, 55, 44, 0, 68, 0, 0, 0, 119),
    "MVS": (47, 99, 0, 0, 67, 0, 0, 0, 119),
    "BV": (137, 0, 3, 4, 29, 19, 111, 1, 0),
  ),
  
)

#show table.cell.where(x: 0): strong
#show table.cell.where(y: 0): strong
#show table.cell: set text(size: 0.9em)
#figure(
  table(
    columns: 13,
    column-gutter: (auto, 3pt, auto, auto, 2pt, auto, auto, 2pt, auto, auto, 3pt, auto),
    row-gutter: (auto, 3pt, auto, auto, 2pt, auto, auto, 2pt, auto, auto, 2pt, auto, auto, 2pt, auto, auto),
    table.header(
      table.cell("Clustering Method", rowspan: 2, align: horizon),
      table.cell("Scenario", rowspan: 2, align: horizon),
      table.cell("Adélie", colspan: 3),
      table.cell("Chinstrap", colspan: 3),
      table.cell("Gentoo", colspan: 3),
      table.cell("Total", colspan: 2),
      [H], [FP], [FN],
      [H], [FP], [FN],
      [H], [FP], [FN],
      [H], [M],
    ),
    ..data.pairs().map(((cluster, scenarios)) => (
      table.cell(cluster, rowspan: scenarios.len(), align: horizon),
      ..scenarios.pairs().map(((scenario, matrix)) => (
        scenario, 
        [#matrix.at(0)],
        str(matrix.at(3) + matrix.at(6)),
        str(matrix.at(1) + matrix.at(2)),
        [#matrix.at(4)],
        str(matrix.at(1) + matrix.at(7)),
        str(matrix.at(3) + matrix.at(5)),
        [#matrix.at(8)],
        str(matrix.at(2) + matrix.at(5)),
        str(matrix.at(6) + matrix.at(7)),
        
        str(matrix.at(0) + matrix.at(4) + matrix.at(8)),
        str(matrix.at(1) + matrix.at(2) + matrix.at(3) + matrix.at(5) + matrix.at(6) + matrix.at(7)),
      ))
      
    )).flatten()
  ),
  caption: flex-caption[Classification results for every clustering method and scenario by species. Scenarios: MV (MultiVariate), MVS (MultiVariate Scaled), and BV (BiVariate). Columns: H (Hits), M (Misses), FP (False Positives), FN (False Negatives).][Classification results for every clustering method and scenario by species.]
) <table:results>

#figure(
  image("images/times_comparison.png", width: 70%),
  caption: [Results of computational times for each clustering method.]
) <times:computational_times>

@times:computational_times shows the execution time of the different clustering methods for datasets of different size. These datasets have been generated automatically based on the Palmer Penguins Dataset, keeping the distribution of each feature (although not the correlations).

As we can see in @times:computational_times, the computational time for exact K-Medoids grows exponentially. When $n$ gets larger, the running times become very high, showing the scalability problems of the exact method. In practice, this makes heuristic methods or K-Means much more practical. K-Means has linear computational cost, and the heuristic K-Medoids implementation saves a lot of time, behaving almost like a linear method in practice.

Regarding MST methods, MST-H is clearly the slowest compared to the heuristics, and even if it can sometimes be faster than exact K-Medoids, its combination of clustering quality and computational cost makes it the least favorable choice overall. It is important to note that Normal MST is excluded from the comparison, as its results were not reliable enough to be considered a valid method. Only the multivariate version produced somewhat acceptable results, but they were far from sufficient. Given this, we can conclude that both Normal MST and MST-H represent the worst options for clustering.

Among all the methods explored, K-Means stands out as the best, providing a good balance between clustering quality and computational efficiency. It consistently gives accurate clusters while staying very fast, even as the dataset grows. For K-Medoids, the heuristic version is also a reasonable choice, since it seems to be as good as the exact method but can handle much larger datasets.

Finally, is important to note that an important factor affecting clustering performance is data preprocessing and variable weighting. When the dataset is balanced and more importance is given to the most relevant variables, all clustering methods improve significantly. This happens because:

- *Weighting important variables*: Amplifies meaningful differences between data points. Variables with high relevance contribute more to the distance or similarity metrics that drive clustering. 

- *Noise reduction*: Giving less importance to irrelevant or noisy variables reduces their influence, preventing the algorithm from being “distracted” by dimensions that do not provide useful information.

This also explains why in K-Medoids and K-Means using bivariate or the scaled version we get better results than the multivariate one, as they focus on the most relevant features of the dataset. 


#set text(size: 0.9em)
#bibliography("ref.bib")
