"""This module contains methods commonly used for preprocessing of data.

The module contains the class DataPreprocessor.

Typical usage example:

    dp = data_preprocessor.DataPreprocessor(config)
"""
import os
import copy
import pickle
import collections
import numpy
import scipy
import pandas
import sklearn.preprocessing
import sklearn.model_selection


class DataPreprocessor:
    """Combines methods commonly used for preprocessing of data.

    Three methods form its api: load_data(), preprocess_data() and save_preprocessed_data(). They
    define the usual workflow with this class. The exact configuration of the preprocessing has
    to be set within these three methods. Editing method calls and reangarring them is encouraged.

    Attributes:
        _config:            Dictionary with basic configuration variables.

        _df:                A pandas dataframe containing the loaded and progressively preprocessed 
                            samples.
        _unscaled_split_df: A dictionary of pandas dataframes containing the unscaled split samples
                            (training, validation, testing).
        _scaled_split_df:   A dictionary of pandas dataframes containing the scaled split samples
                            (training, validation, testing).
        _scaler:            A sklearn standard scaler instance for scaling of the data.
        _scaler_file:       A string containing the name of the pickle file to which the scaler
                            object is serialized.

        _z_threshold:       Parameter of preprocessing, this instance variable here is only used
                            for saving an info later. Do not set the parameter here.
        _outlier_strategy:  Parameter of preprocessing, this instance variable here is only used
                            for saving an info later. Do not set the parameter here.
        _split_sizes:       Parameter of preprocessing, this instance variable here is only used
                            for saving an info later. Do not set the parameter here.
    """
    def __init__(self, config):
        """Initializes the instance with basic configuration variabls and declares important
           instance variables.
        Args:
            config: Dictionary with basic configuration variables
        """
        self._config = config

        self._df = None
        self._split_indices     = {}
        self._unscaled_split_df = {}
        self._scaled_split_df   = {}
        self._scaler            = None
        self._scaler_file       = "scaler"

        # Instance variables for writing info file later - set parameters in preprocess_data()
        self._z_threshold      = None
        self._outlier_strategy = None
        self._split_sizes      = None
    
    
    def run_raw_preprocess(self):
        self.load_data()
        self.preprocess_data()
        self.save_preprocessed_data()


    def load_data(self):
        """Loads the data into a pandas dataframe.

        Either loads all the samples contained in one input file into a pandas dataframe or - if
        the inputs are already split - loads the split input files and temporarily combines the
        contained samples into an equivalent pandas dataframe.
        """
        if self._config["input_already_split"]:
            self._df = self._load_already_split_data()
        else:
            self._df = self._read_data(self._config["data_file_name"])

        self._print_loaded_data()


    def preprocess_data(self):
        """Performs the actual preprocessing of the data.

        Contains methods commonly used for data preprocessing, such as removal of gaps and
        duplicates or scaling. The order in which the methods are applied should only be changed
        after careful checking of dependencies. The parameters of the methods are set herein, it is
        therefore encouraged to change the parameters defined in the method calls.
        """
        self._check_gaps()
        self._check_duplicates()
        self._print_cleaned_data()

        #self._one_hot_encode_features(column_name = "Sex")
        # self._ordinal_encode_features(column_name = "", categories = ["", ""]) # Encoding: categories = [0, 1, 2, ...]
        #self._encode_labels(column_name = "Sex", classes = ["M", "F", "I"]) # Encoding: classes = [0, 1, 2, ...]

        self._reorder_columns(new_order = None) # New order of columns has to be given as list. Do not include the index-column here
        self._delete_columns(columns = None) # Columns to be deleted have to be given as list.

        # Handling of outliers: _handle_outliers_inclusive() checks all columns for outliers that are given as list or by default none.
        #                       _handle_outliers_exclusive() checks all columns except the ones given as list for outliers or by default all.
        # z_threshold: Number of standard deviations by which a value may deviate from the mean of the series
        self._handle_outliers_exclusive(z_threshold = 3, exclude_columns = None)

        # Splitting: If val_size=0.0 or test_size=0.0, the corresponding dataset is not created.
        #            With stratified=True, the relative frequencies of class labels are preserved.
        #            In this case, stratify_column must be provided.
        self._split_data()

        # Scaling: _scale_data_inclusive() scales all columns given as list or by default none.
        #          _scale_data_exclusive() scales all columns except the ones given as list or by default all.
        # Data are scaled to zero mean and unit variance, which is generally a good first approach, both for ReLU and tanh. Categorical columns usually do not have to be scaled.
        self._scale_data_exclusive(exclude_columns = None)


    def save_preprocessed_data(self):
        """Saves the obtained results.

        First, this function clears the output directory. Then, the unscaled and scaled split
        dataframes are saved. The scaler is exported to be reloaded by other programs to rescale
        the data. Lastly, an info file is created summarizing the performed preprocessing for
        documentation.
        """
        self._clear_output_dir()
        self._save_split_dfs(write_index = False) # xn and yn may be set here (separately) explicitly, otherwise only one yn is assumed.
        self._export_scaler()
        self._save_info()


    def _read_data(self, file_name):
        return pandas.read_csv(filepath_or_buffer = os.path.join(self._config["input_dir"], file_name),
                                delimiter        = ",",
                                na_values        = "?",
                                # names            = [f"x{idx}" for idx in range(5)],
                                #index_col        = 0,
                                skipinitialspace = True,
                                skiprows         = 0
        )


    def _load_already_split_data(self):
        split_dfs         = {}
        split_start_index = 0

        for split, split_data_file_name in self._config["split_data_file_names"].items():
            split_dfs[split] = self._read_data(split_data_file_name)

        df = pandas.concat(
            objs         = list(split_dfs.values()),
            axis         = "index",
            ignore_index = True
        )

        for split, split_df in split_dfs.items():
            split_end_index            = split_start_index + split_df.shape[0]
            self._split_indices[split] = df.index[split_start_index:split_end_index]
            split_start_index          = split_end_index

        return df


    def _check_gaps(self):
        self._df.dropna(inplace=True)


    def _check_duplicates(self):
        self._df.drop_duplicates(inplace=True)


    def _one_hot_encode_features(self, column_name):
        column_idx = self._df.columns.get_loc(column_name)

        oh_columns = pandas.get_dummies(self._df[column_name], dtype=int, prefix=f"{column_name}_oh")

        if column_idx == 0:                 # Column to be one-hot-encoded is first column
            right_df = self._df.drop(columns=column_name)
            self._df = pandas.concat([oh_columns, right_df], axis="columns")
            return

        if column_idx == self._df.shape[1]: # Column to be one-hot-encoded is last column
            left_df = self._df.drop(columns=column_name)
            self._df = pandas.concat([left_df, oh_columns], axis="columns")
            return

                                            # Column to be one-hot-encoded is neither first nor last
        left_df  = self._df.loc[:,:column_name].drop(columns=column_name)
        right_df = self._df.loc[:,column_name:].drop(columns=column_name)
        self._df = pandas.concat([left_df, oh_columns, right_df], axis="columns")
        return


    def _ordinal_encode_features(self, column_name, categories):
        oe = sklearn.preprocessing.OrdinalEncoder(categories=[categories])
        self._df[[column_name]] = oe.fit_transform(self._df[[column_name]])


    def _encode_labels(self, column_name, classes):
        le = sklearn.preprocessing.LabelEncoder().fit(classes)
        self._df[column_name] = le.transform(self._df[column_name])


    def _reorder_columns(self, new_order=None):
        if new_order is None:
            return
        self._df = self._df.reindex(columns=new_order, copy=False)


    def _delete_columns(self, columns=None):
        if columns is None:
            return
        self._df.drop(columns=columns, inplace=True)


    def _handle_outliers_inclusive(self, z_threshold, include_columns=None): # By default (include_columns=None), no columns are checked for outliers
        if include_columns is None:
            self._print_outliers_separator()
            print("Outliers are ignored.\n")
            self._outlier_strategy = "ignore"
        else:
            self._handle_outliers_internal(z_threshold, self._df.columns.drop(include_columns))


    def _handle_outliers_exclusive(self, z_threshold, exclude_columns=None): # By default (exclude_columns=None), all columns are checked for outliers
        if exclude_columns is None:
            self._handle_outliers_internal(z_threshold, [])
        else:
            self._handle_outliers_internal(z_threshold, exclude_columns)


    def _handle_outliers_internal(self, z_threshold, exclude_columns): # Internal implementation - do not call!
        self._z_threshold = z_threshold # Save for info file

        mean = self._df.mean()
        std = self._df.std()

        outliers, clip_values, z_scores = self._find_outliers(z_threshold, mean, std, exclude_columns)

        self._print_outliers(outliers, clip_values, z_scores, mean, std)

        strategy = self._select_outlier_strategy()
        self._outlier_strategy = strategy # Save for info file

        if strategy == "keep":
            pass
        elif strategy == "clip":
            self._clip_outliers(outliers, clip_values)
        elif strategy == "remove":
            self._delete_outliers(outliers)


    def _find_outliers(self, z_threshold, mean, std, exclude_columns):
        z_scores = numpy.abs(scipy.stats.zscore(self._df)) # Z-score (by how many standard deviations a value deviates from the mean of the series)
                                                           # is computed for each value in the dataframe

        outlier_coordinates = numpy.where(z_scores.to_numpy() > z_threshold)

        outliers    = collections.defaultdict(list) # Structure: Keys are all row indices with outliers, values for each key are the column indices of the outliers
        clip_values = collections.defaultdict(list) # Strucutre: Keys are all row indices with outliers, values for each key are the values the data would be clipped to

        for x, y in zip(*outlier_coordinates):
            row_idx     = z_scores.index[x]
            column_idx  = y
            column_name = self._df.columns[column_idx]

            if column_name in exclude_columns:
                continue

            clip_value = self._compute_clip_value(row_idx, column_name, z_threshold, mean, std)

            outliers[row_idx].append(column_idx)
            clip_values[row_idx].append(clip_value)

        return outliers, clip_values, z_scores


    def _compute_clip_value(self, row_idx, column_name, z_threshold, mean, std):
        current_value = self._df.at[row_idx, column_name]
        mean_value    = mean[column_name]
        std_value     = std[column_name]

        if current_value.any() < mean_value:
            clip_value = mean_value - z_threshold*std_value
        else:
            clip_value = mean_value + z_threshold*std_value

        if current_value.dtype == "int32" or current_value.dtype == "int64":
            return clip_value
        if current_value.dtype == "float64":
            return int(round(clip_value))

        raise TypeError("Encountered unexpected data type while handling outliers, expect either int32, int64, or float64.")


    def _print_outliers(self, outliers, clip_values, z_scores, mean, std):
        self._print_outliers_separator()

        for outlier_counter, ((row_idx, column_indices), (_, column_clip_values)) in enumerate(zip(outliers.items(), clip_values.items()), start=1):

            print(f"\n- #{outlier_counter}: Sample {str(row_idx)} with outlier(s): -")
            columns_split = self._df.loc[row_idx].to_string().split("\n")

            for column_idx, column in enumerate(columns_split):

                if column_idx in column_indices:
                    self._print_outlier(column, row_idx, self._df.columns[column_idx], column_clip_values[column_indices.index(column_idx)], z_scores, mean, std)
                else:
                    self._print_non_outlier(column)


    def _print_outlier(self, column, row_idx, column_name, clip_value, z_scores, mean, std):
        val_z_score = z_scores.at[row_idx, column_name]
        val_mean    = mean[column_name]
        val_std     = std[column_name]
        separator   = "#" * len(column) + "##"
        print(separator)
        print(f"{column} - Z-score: {self._to_str(val_z_score)} - Mean: {self._to_str(val_mean)} - Std: {self._to_str(val_std)} - Clip: {self._to_str(clip_value)}")
        print(separator)


    def _to_str(self, num):
        return str(round(num, 3))


    def _print_non_outlier(self, column):
        print(column)


    def _select_outlier_strategy(self):
        while True:
            strategy = self._ask_for_outlier_strategy()
            if self._confirm_outlier_strategy(strategy):
                return strategy


    def _ask_for_outlier_strategy(self):
        while True:
            user_input = input("Type 'k' to keep outliers, 'c' to clip outliers and 'r' to remove outliers:").lower()
            if user_input == "k":
                return "keep"
            if user_input == "c":
                return "clip"
            if user_input == "r":
                return "remove"


    def _confirm_outlier_strategy(self, strategy):
        user_input = input(f"You have selected to {strategy} outliers. Confirm with 'y', cancel with 'n':").lower()
        return user_input == "y"


    def _clip_outliers(self, outliers, clip_values):
        for (row_idx, column_indices), (_, column_clip_values) in zip(outliers.items(), clip_values.items()):
            for column_idx, clip_value in zip(column_indices, column_clip_values):
                self._df.at[row_idx, self._df.columns[column_idx]] = clip_value


    def _delete_outliers(self, outliers):
        self._df.drop(index=outliers.keys(), inplace=True)


    def _split_data(self, val_size=0.15, test_size=0.15, stratified=False, stratify_column=None):
        self._split_sizes = [1.0-val_size-test_size, val_size, test_size] # Save for info file

        if self._config["input_already_split"]:
            self._re_split_data()

        elif stratified:
            self._split_data_stratified(val_size, test_size, stratify_column)

        else: # unstratified
            self._split_data_unstratified(val_size, test_size)


    def _re_split_data(self): # Restore original data split in case one was already provided
        for split, split_indices in self._split_indices.items():
            remaining_indices = [idx in self._df.index for idx in split_indices]
            self._unscaled_split_df[split] = self._df.loc[split_indices[remaining_indices]]


    def _split_data_stratified(self, val_size, test_size, stratify_column): # Stratify preserves relative class frequencies in split datasets
        if val_size == 0 and test_size == 0:
            self._unscaled_split_df["training"] = self._df
            return

        if val_size == 0:
            self._unscaled_split_df["training"], self._unscaled_split_df["test"], = sklearn.model_selection.train_test_split(self._df, test_size=test_size, stratify=self._df[stratify_column], random_state=42)
            return

        if test_size == 0:
            self._unscaled_split_df["training"], self._unscaled_split_df["validation"], = sklearn.model_selection.train_test_split(self._df, test_size=val_size, stratify=self._df[stratify_column], random_state=42)
            return

        self._unscaled_split_df["training"], temp_df, _, temp_label = sklearn.model_selection.train_test_split(self._df, self._df[stratify_column], test_size=val_size+test_size, stratify=self._df[stratify_column], random_state=42)
        self._unscaled_split_df["validation"], self._unscaled_split_df["test"] = sklearn.model_selection.train_test_split(temp_df, test_size=test_size/(val_size+test_size), stratify=temp_label, random_state=42)
        return


    def _split_data_unstratified(self, val_size, test_size):
        if val_size == 0 and test_size == 0:
            self._unscaled_split_df["training"] = self._df
            return

        if val_size == 0:
            self._unscaled_split_df["training"], self._unscaled_split_df["test"], = sklearn.model_selection.train_test_split(self._df, test_size=test_size, random_state=42)
            return

        if test_size == 0:
            self._unscaled_split_df["training"], self._unscaled_split_df["validation"], = sklearn.model_selection.train_test_split(self._df, test_size=val_size, random_state=42)
            return

        self._unscaled_split_df["training"], temp_df = sklearn.model_selection.train_test_split(self._df, test_size=val_size+test_size, random_state=42)
        self._unscaled_split_df["validation"], self._unscaled_split_df["test"] = sklearn.model_selection.train_test_split(temp_df, test_size=test_size/(val_size+test_size), random_state=42)
        return


    def _scale_data_inclusive(self, include_columns=None): # By default (include_columns=None), no columns are scaled
        if include_columns is None:
            self._scale_data_internal(columns = self._unscaled_split_df["training"].columns, do_scale = False)
        else:
            self._scale_data_internal(columns = include_columns, do_scale = True)


    def _scale_data_exclusive(self, exclude_columns=None): # By default (exclude_columns=None), all columns are scaled
        if exclude_columns is None:
            self._scale_data_internal(columns = self._unscaled_split_df["training"].columns, do_scale = True)
        else:
            self._scale_data_internal(columns = self._unscaled_split_df["training"].columns.drop(exclude_columns), do_scale = True)


    def _scale_data_internal(self, columns, do_scale): # Internal implementation - do not call!
        if do_scale:
            self._scaler = sklearn.preprocessing.StandardScaler()
        else:
            self._scaler = sklearn.preprocessing.StandardScaler(with_mean=False, with_std=False)

        self._scaled_split_df = copy.deepcopy(self._unscaled_split_df)
        self._scaler = self._scaler.fit(self._scaled_split_df["training"][columns])

        for _, scaled_split in self._scaled_split_df.items():
            scaled_split[columns] = self._scaler.transform(scaled_split[columns])


    def _clear_output_dir(self):
        for file in os.listdir(self._config["output_dir"]):
            os.remove(os.path.join(self._config["output_dir"], file))


    def _save_split_dfs(self, xn=None, yn=None, write_index=False):
        xn, yn = self._set_xn_yn(xn, yn)
        self._save_split_df(self._unscaled_split_df, xn, yn, write_index, "unscaled")
        self._save_split_df( self._scaled_split_df, xn, yn, write_index,  "scaled")


    def _set_xn_yn(self, xn, yn):
        if xn is None and yn is None:
            return self._scaled_split_df["training"].shape[1]-1, 1
        if xn is None:
            return self._scaled_split_df["training"].shape[1]-yn, yn
        if yn is None:
            return xn, self._scaled_split_df["training"].shape[1]-xn

        return xn, yn


    def _save_split_df(self, split_dfs, xn, yn, write_index, name_extension):
        for split, split_df in split_dfs.items():
            with open(os.path.join(self._config["output_dir"], f"{name_extension}_{split}_{self._config['output_data_file_name']}.csv"), "w", newline="") as out_file:
                out_file.write(f"xn={str(xn)},yn={str(yn)}\n")             # Header
                split_df.to_csv(out_file, header=False, index=write_index) # Data


    def _export_scaler(self):
        with open(os.path.join(self._config["output_dir"], f"{self._scaler_file}.pickle"), "wb") as out_file:
            pickle.dump(obj=self._scaler, file=out_file, protocol=pickle.HIGHEST_PROTOCOL)


    def _save_info(self):
        with open(os.path.join(self._config["output_dir"], f"{self._config['info_file_name']}.txt"), "w") as out_file:
            out_file.write("- Info: -\n")

            totalNumSamples = sum([scaled_split_df.shape[0] for _, scaled_split_df in self._scaled_split_df.items()])
            out_file.write(f"\nTotal number of samples: {totalNumSamples}")

            for split, scaled_split_df in self._scaled_split_df.items():
                out_file.write(f"\nNumber of {split} samples: {scaled_split_df.shape[0]}")

            if self._config["input_already_split"]:
                out_file.write("\nSplit sizes: Preserved split provided by the input data files.")
            else:
                out_file.write(f"\nSplit sizes: {round(self._split_sizes[0], 2)} / {round(self._split_sizes[1], 2)} / {round(self._split_sizes[2], 2)} (Training / Validation / Testing)")

            out_file.write("\n\nColumns:\n")
            for column_idx, column in enumerate(self._scaled_split_df["training"].columns.tolist()):
                out_file.write(f"#{column_idx}: {column}\n")

            out_file.write("\nOutlier handling:\n")
            out_file.write(f"Values are considered outliers if they deviate more than {self._z_threshold} standard deviations from the respective mean.\n")
            out_file.write(f"The selected strategy to handle outliers is to {self._outlier_strategy} them.")


    def _print_loaded_data(self):
        print("###############\n")
        print("### Columns ###\n")
        print("###############\n")
        print(*self._df.columns.tolist(), sep="\n")
        print("\n\n")

        print("#####################\n")
        print("### Original Data ###\n")
        print("#####################\n")
        print(self._df.info())
        print("\n\n")


    def _print_cleaned_data(self):
        print("#############################################\n")
        print("### Data cleaned from gaps and duplicates ###\n")
        print("#############################################\n")
        print(self._df.info())
        print("\n\n")


    def _print_outliers_separator(self):
        print("################\n")
        print("### Outliers ###\n")
        print("################\n")
