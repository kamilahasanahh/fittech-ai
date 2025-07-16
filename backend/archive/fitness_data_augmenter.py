import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

class FitnessDataAugmenter:
    """
    Rule-based synthetic data augmenter for fitness demographic datasets.
    Ensures all nutrition templates are represented, including rare/unnatural ones, using domain logic.
    Fitness goals are forced to be equally distributed in the final training set.
    """
    def __init__(self, df: pd.DataFrame, logger: Optional[logging.Logger] = None):
        self.df = df.copy()
        self.logger = logger or logging.getLogger("FitnessDataAugmenter")
        self.demographic_axes = [
            'age_group', 'gender', 'activity_level', 'bmi_category', 'fitness_goal'
        ]
        self.constraints = {
            'age': (18, 65),
            'height_cm': (150, 200),
            'weight_kg': (40, 150),
        }
        self.bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
        self.fitness_goals = ['Fat Loss', 'Muscle Gain', 'Maintenance']
        self.activity_levels = ['Low Activity', 'Moderate Activity', 'High Activity']
        self.genders = ['Male', 'Female']
        self.age_bins = [18, 30, 45, 65]
        self.age_labels = ['18-30', '31-45', '46-65']
        self.nutrition_template_combinations = [
            # (goal, bmi_category)
            ('Fat Loss', 'Normal'),
            ('Fat Loss', 'Overweight'),
            ('Fat Loss', 'Obese'),
            ('Muscle Gain', 'Underweight'),
            ('Muscle Gain', 'Normal'),
            ('Maintenance', 'Normal'),
            ('Maintenance', 'Overweight'),
        ]
        self._prepare_age_groups()

    def _prepare_age_groups(self):
        if 'age_group' not in self.df.columns:
            self.df['age_group'] = pd.cut(self.df['age'], bins=self.age_bins, labels=self.age_labels, right=True, include_lowest=True)

    def calculate_target_counts(self, max_synth_per_comb=200):
        """
        For each valid nutrition template combination, determine how many synthetic samples are needed.
        Only augment combinations that are valid for templates.
        Args:
            max_synth_per_comb (int): Maximum synthetic samples per combination.
        Returns:
            Dict[Tuple, int]: Mapping from combination tuple to target count.
        """
        # Count real samples for each (goal, bmi_category)
        group_counts = self.df.groupby(['fitness_goal', 'bmi_category']).size()
        target_counts = {}
        for goal, bmi_cat in self.nutrition_template_combinations:
            real_count = group_counts.get((goal, bmi_cat), 0)
            if real_count is None:
                real_count = 0
            real_count = int(real_count)
            # If real_count is 0, we want at least some synthetic samples (e.g., 100)
            if real_count == 0:
                target_counts[(goal, bmi_cat)] = int(min(100, max_synth_per_comb))
            else:
                # Cap synthetic to at most real_count (or max_synth_per_comb)
                target_counts[(goal, bmi_cat)] = int(min(real_count * 2, max_synth_per_comb))
        self.logger.info(f"Target counts per nutrition template combination: {target_counts}")
        return target_counts

    def generate_synthetic_sample_for_template(self, goal, bmi_cat):
        """
        Generate a synthetic sample for a specific (goal, bmi_category) using domain logic.
        """
        # Age: random in valid range
        age = np.random.randint(18, 65)
        gender = np.random.choice(self.genders)
        # Height: normal distribution by gender
        height_cm = np.random.normal(170 if gender == 'Male' else 160, 7)
        height_cm = np.clip(height_cm, 150, 200)
        # BMI: pick a value in the category
        if bmi_cat == 'Underweight':
            bmi = np.random.uniform(16, 18.4)
        elif bmi_cat == 'Normal':
            bmi = np.random.uniform(18.5, 24.9)
        elif bmi_cat == 'Overweight':
            bmi = np.random.uniform(25, 29.9)
        else:
            bmi = np.random.uniform(30, 40)
        weight_kg = bmi * ((height_cm / 100) ** 2)
        weight_kg = np.clip(weight_kg, 40, 150)
        # Activity: random
        activity_level = np.random.choice(self.activity_levels)
        # Mod_act/Vig_act: based on activity_level
        if activity_level == 'High Activity':
            mod_act = np.random.normal(5.8, 0.8)
            vig_act = np.random.normal(2.8, 0.5)
        elif activity_level == 'Moderate Activity':
            mod_act = np.random.normal(3.75, 0.6)
            vig_act = np.random.normal(1.87, 0.3)
        else:
            mod_act = np.random.normal(1.67, 0.4)
            vig_act = np.random.normal(0.83, 0.25)
        mod_act = max(0, mod_act)
        vig_act = max(0, vig_act)
        # Age group
        age_group = pd.cut([age], bins=self.age_bins, labels=self.age_labels, right=True, include_lowest=True)[0]
        # BMR/TDEE (simple formulas)
        bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age) if gender == 'Male' else 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        activity_multipliers = {'Low Activity': 1.29, 'Moderate Activity': 1.55, 'High Activity': 1.81}
        tdee = bmr * activity_multipliers[activity_level]
        # Compose sample
        sample = {
            'age': int(age),
            'gender': gender,
            'height_cm': round(height_cm, 1),
            'weight_kg': round(weight_kg, 1),
            'bmi': round(bmi, 2),
            'bmi_category': bmi_cat,
            'bmr': round(bmr, 1),
            'tdee': round(tdee, 1),
            'activity_level': activity_level,
            'activity_multiplier': activity_multipliers[activity_level],
            'Mod_act': round(mod_act, 2),
            'Vig_act': round(vig_act, 2),
            'fitness_goal': goal,
            'age_group': age_group,
            'data_source': 'synthetic'
        }
        return sample

    def augment_training_data(self, max_synth_per_comb=200) -> pd.DataFrame:
        """
        Augment the dataset to ensure all nutrition template combinations are present and fitness goals are equally distributed.
        Args:
            max_synth_per_comb (int): Maximum synthetic samples per combination.
        Returns:
            pd.DataFrame: Augmented dataset (original + synthetic).
        """
        # Calculate target counts for each nutrition template combination
        target_counts = self.calculate_target_counts(max_synth_per_comb)
        augmented = []
        # For each (goal, bmi_category) combination
        for (goal, bmi_cat), target in target_counts.items():
            mask = (self.df['fitness_goal'] == goal) & (self.df['bmi_category'] == bmi_cat)
            group_df = self.df[mask]
            n_real = int(len(group_df))
            n_needed = int(target - n_real)
            if n_real > 0:
                # Mark all real data as 'real'
                real_df = group_df.copy()
                real_df['data_source'] = 'real'
                augmented.append(real_df)
            if n_needed > 0:
                if n_real > 0:
                    # Interpolate from real data
                    group_df = group_df if isinstance(group_df, pd.DataFrame) else group_df.to_frame().T
                    synth_samples = [self.generate_synthetic_sample(group_df) for _ in range(n_needed)]
                    self.logger.info(f"Generated {n_needed} synthetic samples for ({goal}, {bmi_cat}) from real data.")
                else:
                    synth_samples = [self.generate_synthetic_sample_for_template(goal, bmi_cat) for _ in range(n_needed)]
                    self.logger.warning(f"Generated {n_needed} FULLY SYNTHETIC samples for ({goal}, {bmi_cat}) (no real data).")
                if len(synth_samples) > 0:
                    synth_df = pd.DataFrame(synth_samples)
                    synth_df['data_source'] = 'synthetic'
                    augmented.append(synth_df)
        if not augmented:
            df_aug = self.df.copy()
            if 'data_source' not in df_aug.columns:
                df_aug['data_source'] = 'real'
        else:
            df_aug = pd.concat(augmented, ignore_index=True)
        df_aug = self.validate_synthetic_data(df_aug)
        # Force fitness goals to be equally distributed
        min_goal = int(df_aug['fitness_goal'].value_counts().min())
        balanced = []
        for goal in self.fitness_goals:
            goal_df = df_aug[df_aug['fitness_goal'] == goal]
            if len(goal_df) > min_goal:
                goal_df = goal_df.sample(min_goal, random_state=42)
            balanced.append(goal_df if isinstance(goal_df, pd.DataFrame) else pd.DataFrame([goal_df]))
        df_final = pd.concat(balanced, ignore_index=True)
        return df_final

    def generate_synthetic_sample(self, group_df: pd.DataFrame) -> Dict:
        # Ensure group_df is always a DataFrame
        if isinstance(group_df, pd.Series) or not isinstance(group_df, pd.DataFrame):
            group_df = pd.DataFrame([group_df])
        # If group_df is empty, return a default synthetic sample
        if len(group_df) == 0:
            return self.generate_synthetic_sample_for_template('Maintenance', 'Normal')
        if len(group_df) < 2:
            base = group_df.iloc[0].copy()
            other = base.copy()
        else:
            base, other = group_df.sample(2, replace=True).to_dict('records')
        synth = {}
        for col in ['age', 'height_cm', 'weight_kg']:
            val = np.mean([base[col], other[col]])
            # Ensure bounds are int for np.clip
            min_bound, max_bound = [int(x) for x in self.constraints[col]]
            noise = np.random.uniform(-0.1, 0.1) * val
            synth[col] = np.clip(val + noise, min_bound, max_bound)
        synth['bmi'] = synth['weight_kg'] / ((synth['height_cm'] / 100) ** 2)
        synth['bmi_category'] = self._categorize_bmi(synth['bmi'])
        for col in ['gender', 'activity_level', 'fitness_goal', 'age_group']:
            synth[col] = base[col]
        for col in ['Mod_act', 'Vig_act']:
            if col in group_df.columns:
                base_val = base.get(col, 0)
                other_val = other.get(col, 0)
                val = np.mean([base_val, other_val])
                noise = np.random.uniform(-0.1, 0.1) * val
                synth[col] = max(0.0, float(val + noise))
        for col in group_df.columns:
            if col not in synth:
                if pd.api.types.is_numeric_dtype(group_df[col]):
                    synth[col] = np.mean([base.get(col, 0), other.get(col, 0)])
                else:
                    synth[col] = base.get(col, None)
        synth['data_source'] = 'synthetic'
        return synth

    def validate_synthetic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (min_v, max_v) in self.constraints.items():
            df[col] = df[col].clip(min_v, max_v)
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        df['bmi_category'] = df['bmi'].apply(self._categorize_bmi)
        valid_mask = df.apply(self._is_valid_combination, axis=1)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            self.logger.warning(f"Removed {n_invalid} invalid synthetic samples.")
        df = df[valid_mask].reset_index(drop=True)
        return df

    def _categorize_bmi(self, bmi: float) -> str:
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

    def _is_valid_combination(self, row) -> bool:
        # No muscle gain for obese, no fat loss for underweight, etc.
        if row['bmi_category'] == 'Underweight' and row['fitness_goal'] == 'Fat Loss':
            return False
        if row['bmi_category'] == 'Obese' and row['fitness_goal'] == 'Muscle Gain':
            return False
        return True

    def _bin_height(self, df):
        # Bin height into 3 groups: 150-165, 166-180, 181-200
        bins = [150, 165, 180, 200]
        labels = ['150-165', '166-180', '181-200']
        df['height_bin'] = pd.cut(df['height_cm'], bins=bins, labels=labels, right=True, include_lowest=True)
        return df

    def _bin_weight(self, df):
        # Bin weight into 3 groups: 40-70, 71-100, 101-150
        bins = [40, 70, 100, 150]
        labels = ['40-70', '71-100', '101-150']
        df['weight_bin'] = pd.cut(df['weight_kg'], bins=bins, labels=labels, right=True, include_lowest=True)
        return df

    def multi_factor_balance(self, min_per_comb=10, max_per_comb=200, random_state=42):
        """
        Balance the training set across all combinations of age_group, gender, height_bin, weight_bin, activity_level, and fitness_goal.
        Uses both over- and under-sampling as needed. Returns the balanced DataFrame.
        """
        np.random.seed(random_state)
        df = self.df.copy()
        # Ensure bins exist
        if 'age_group' not in df.columns:
            age_bins = [18, 30, 45, 65]
            age_labels = ['18-30', '31-45', '46-65']
            df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)
        df = self._bin_height(df)
        df = self._bin_weight(df)
        group_cols = ['age_group', 'gender', 'height_bin', 'weight_bin', 'activity_level', 'fitness_goal']
        grouped = df.groupby(group_cols)
        # Find the target count for each group (max for oversampling, min for undersampling, within limits)
        group_sizes = grouped.size()
        max_group_size = int(group_sizes.max())
        target = min(max(max_group_size, int(min_per_comb)), int(max_per_comb))
        balanced = []
        for comb, group in grouped:
            if isinstance(group, pd.Series):
                group = group.to_frame().T
            n = len(group)
            if n == 0:
                continue
            if n < target:
                # Oversample with replacement
                sampled = group.sample(target, replace=True, random_state=random_state)
            elif n > target:
                # Undersample
                sampled = group.sample(target, replace=False, random_state=random_state)
            else:
                sampled = group
            balanced.append(sampled)
        df_balanced = pd.concat(balanced, ignore_index=True)
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return df_balanced

    @staticmethod
    def concat_and_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(df)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train = df.iloc[:n_train].copy()
        val = df.iloc[n_train:n_train+n_val].copy()
        test = df.iloc[n_train+n_val:].copy()
        train['split'] = 'train'
        val['split'] = 'validation'
        test['split'] = 'test'
        return train, val, test 

    def augment_training_data_balanced(self, target_per_comb=None):
        group_cols = ['fitness_goal', 'activity_level', 'bmi_category']
        real_counts = self.df.groupby(group_cols).size()
        if target_per_comb is None:
            target_per_comb = real_counts.max()
        augmented = []
        for combo, real_count in real_counts.items():
            group_df = self.df[
                (self.df['fitness_goal'] == combo[0]) &
                (self.df['activity_level'] == combo[1]) &
                (self.df['bmi_category'] == combo[2])
            ]
            n_needed = int(target_per_comb - real_count)
            # Always include all real data
            augmented.append(group_df)
            if n_needed > 0:
                synth_samples = [self.generate_synthetic_sample(group_df) for _ in range(n_needed)]
                if synth_samples:
                    synth_df = pd.DataFrame(synth_samples)
                    synth_df['data_source'] = 'synthetic'
                    augmented.append(synth_df)
            elif n_needed < 0:
                # Downsample if too many (optional)
                group_df = group_df.sample(target_per_comb, random_state=42)
                augmented[-1] = group_df
        df_aug = pd.concat(augmented, ignore_index=True)
        return df_aug 

    def augment_training_data_strict_combinations(self, valid_combinations, target_per_comb=500):
        augmented = []
        for combo in valid_combinations:
            goal, activity, bmi_cat = combo
            group_df = self.df[
                (self.df['fitness_goal'] == goal) &
                (self.df['activity_level'] == activity) &
                (self.df['bmi_category'] == bmi_cat)
            ]
            real_count = len(group_df)
            # If more real samples than target, downsample
            if real_count >= target_per_comb:
                sampled_real = group_df.sample(target_per_comb, random_state=42)
                augmented.append(sampled_real)
            else:
                # Use all real, add synthetic to reach target
                if real_count > 0:
                    augmented.append(group_df)
                    n_needed = target_per_comb - real_count
                    synth_samples = [self.generate_synthetic_sample(group_df) for _ in range(n_needed)]
                else:
                    n_needed = target_per_comb
                    synth_samples = [self.generate_synthetic_sample_for_template(goal, bmi_cat) for _ in range(n_needed)]
                if synth_samples:
                    synth_df = pd.DataFrame(synth_samples)
                    synth_df['data_source'] = 'synthetic'
                    augmented.append(synth_df)
        df_aug = pd.concat(augmented, ignore_index=True)
        return df_aug 