
import pandas as pd
import numpy as np

class CleanedData:

  
    @staticmethod
    def fill_homeplanet(df):

        df['HomePlanet'] = df.groupby('Group')['HomePlanet'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
        )

        def fill_strong_decks(row):
            if pd.isna(row['HomePlanet']):
                if row['Deck'] in ['A', 'B', 'C', 'T']:
                    return 'Europa'
                elif row['Deck'] == 'G':
                    return 'Earth'
            return row['HomePlanet']

        df['HomePlanet'] = df.apply(fill_strong_decks, axis=1)

        df['HomePlanet'] = df.groupby('Deck')['HomePlanet'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
        )

        dest_to_planet = df.dropna(subset=['HomePlanet']) \
            .groupby('Destination')['HomePlanet'] \
            .agg(lambda x: x.value_counts().index[0])

        df.loc[df['HomePlanet'].isna(), 'HomePlanet'] = \
            df.loc[df['HomePlanet'].isna(), 'Destination'].map(dest_to_planet)

        df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)

        return df


    @staticmethod
    def cleaning_df(df):

        df['Group'] = df['PassengerId'].str.split('_').str[0]

        df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)

        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df[cols] = df[cols].fillna(0)

        df['TotalSpend'] = df[cols].sum(axis=1)

        df.loc[df['CryoSleep'] == True, cols] = 0

        # FIXED CALL (IMPORTANT)
        df = CleanedData.fill_homeplanet(df)

        df.loc[(df['CryoSleep'].isna()) & (df['TotalSpend'] == 0), 'CryoSleep'] = True
        df.loc[(df['CryoSleep'].isna()) & (df['TotalSpend'] > 0), 'CryoSleep'] = False

        df.loc[df['Age'] == 0, 'Age'] = None

        df['Age'] = df.groupby(['HomePlanet', 'CryoSleep'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )

        df['Deck'] = df.groupby('Group')['Deck'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
        )

        df['Deck'] = df.groupby('HomePlanet')['Deck'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
        )

        df['Side'] = df.groupby('Group')['Side'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
        )

        df['Side'].fillna(df['Side'].mode()[0], inplace=True)

        home_to_dest = df.groupby('HomePlanet')['Destination'] \
            .agg(lambda x: x.value_counts().index[0])

        df.loc[df['Destination'].isna(), 'Destination'] = \
            df.loc[df['Destination'].isna(), 'HomePlanet'].map(home_to_dest)

        df.loc[df['VIP'].isna() & (df['TotalSpend'] > df['TotalSpend'].median()), 'VIP'] = True
        df.loc[df['VIP'].isna() & (df['TotalSpend'] <= df['TotalSpend'].median()), 'VIP'] = False

        df.drop(['Cabin','CabinNum'], axis=1, inplace=True)

        return df