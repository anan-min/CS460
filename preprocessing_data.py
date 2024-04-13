from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

paths = ['data/reviews_0-250.csv',
         'data/reviews_250-500.csv',
         'data/reviews_500-750.csv',
         'data/reviews_750-1250.csv',
         'data/reviews_1250-end.csv']


for path in paths:
    df = pd.read_csv(path, low_memory=False)
    # Perform preprocessing steps here
    rating_mean = df['rating'].mean()
    helpfulness_mean = df['helpfulness'].mean()
    is_recommend_mode = df['is_recommended'].mode()[0]
    skin_tone_mode = df['skin_tone'].mode()[0]
    eye_color_mode = df['eye_color'].mode()[0]
    skin_type_mode = df['skin_type'].mode()[0]
    hair_color_mode = df['hair_color'].mode()[0]

    df['product_id'].astype(str)
    df['rating'] = df['rating'].fillna(rating_mean)
    df['helpfulness'] = df['helpfulness'].fillna(helpfulness_mean)
    df['is_recommended'] = df['is_recommended'].fillna(is_recommend_mode)
    df['skin_tone'] = df['skin_tone'].fillna(skin_tone_mode)
    df['eye_color'] = df['eye_color'].fillna(eye_color_mode)
    df['skin_type'] = df['skin_type'].fillna(skin_type_mode)
    df['hair_color'] = df['hair_color'].fillna(hair_color_mode)
    df['product_id'] = df['product_id'].str.replace('P', '').astype(int)
    df.drop(['review_text', 'review_title', 'product_name',
             'brand_name', 'submission_time'], axis=1, inplace=True)

    # label_encoder = LabelEncoder()
    # df['author_id_encoded'] = label_encoder.fit_transform(df['author_id'])
    df.drop('author_id', axis=1, inplace=True)

    cat_cols = ['author_id_encoded', 'skin_tone', 'eye_color', 'skin_type',
                'hair_color', 'product_id', 'product_name', 'brand_name']
    date_cols = ['submission_time']
    num_cols = ['rating', 'is_recommended', 'helpfulness', 'total_feedback_count',
                'total_neg_feedback_count', 'total_pos_feedback_count', 'price_usd']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols),
            ('date', 'passthrough', date_cols)
        ])

    df_encoded = pd.get_dummies(
        df, columns=['skin_tone', 'eye_color', 'skin_type', 'hair_color'])

    # Output the processed DataFrame into a new CSV file
    output_path = f'{path.split(".")[0]}_processed.csv'
    df_encoded.to_csv(output_path, index=False)
    print(f"Processed CSV saved at: {output_path}")



