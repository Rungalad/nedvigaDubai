import streamlit as st
import pickle
import pandas as pd
import copy 

from typing import Dict, List, Set

cols = pickle.load(open(r"multiselect.pkl", 'rb'))
keys = list(cols.keys())

@st.cache_data
def load_full_dataset(path: str = r"dataset_from_2023.pkl") -> pd.DataFrame:
    dataset = pd.read_pickle(path)
    return dataset

@st.cache_data
def load_data(path: str = r"all_combs_of_filters.pkl") -> pd.DataFrame:
    all_combs = pd.read_pickle(path)
    all_combs['selected'] = True
    return all_combs
    
@st.cache_resource 
def load_model():
    return pickle.load(open(r"catboost_it100_dp16.pkl", 'rb'))

def initialize_state():
    """
    Initializes all filters and counter in Streamlit Session State
    """
    for subkey in keys:
        if subkey not in st.session_state:
            st.session_state[subkey] = set()
    if "counter" not in st.session_state:
        st.session_state.counter = 0

def reset_state_callback():
    """
    Resets all filters and increments counter in Streamlit Session State
    """
    st.session_state.counter = 1 + st.session_state.counter

    for subkey in keys:
        st.session_state[subkey] = set()
    
def update_state(current_query: Dict[str, Set]):
    """
    Stores input dict of filters into Streamlit Session State.

    If one of the input filters is different from previous value in Session State, 
    rerun Streamlit to activate the filtering and plot updating with the new info in State.
    """
    rerun = False
    for subkey in keys:
        if current_query[subkey] - st.session_state[subkey]:
            st.session_state[subkey] = current_query[subkey]
            rerun = True
    if rerun:
        st.rerun()
        
def query_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filters in Streamlit Session State
    to filter the input DataFrame
    """
    for subkey in keys:
        if st.session_state[subkey]:
            df.loc[~df[subkey].isin(st.session_state[subkey]), "selected"] = False
    return df

def get_ui(cols, keys, tranformed_data, full_dataset, model):
    """
    Get UI
    """
    st.header("Расчет стоимости квартиры")
    get_di = {}
    lens_di = {}
    prom = tranformed_data[tranformed_data["selected"] == True]
    for col in prom:
        get_di.update({col: set(prom[col].values)})
        lens_di.update({col: len(set(prom[col].values))})
        
    opts = {}
    st.subheader("Расположение объекта")
    main_cols = ['area_name_en', 'building_name_en', 'project_name_en', 'master_project_en']
    for key, col in zip(main_cols, st.columns(len(main_cols), gap='large')):
        ln = lens_di[key]
        with col:
            options = st.multiselect(
                f"Выберите {key} (осталось вариантов {ln}):",
                list(get_di[key]),
                list(st.session_state[key]),
                placeholder="Выберите опцию")
            opts.update({key: set(options)})

    st.subheader("Объекты рядом")
    near_cols = ['nearest_landmark_en','nearest_metro_en', 'nearest_mall_en']
    for key, col in zip(near_cols, st.columns(len(near_cols), gap='large')):
        ln = lens_di[key]
        with col:
            options = st.multiselect(
                f"Выберите {key} (осталось вариантов {ln}):",
                list(get_di[key]),
                list(st.session_state[key]),
                placeholder="Выберите опцию")
            opts.update({key: set(options)})
    
    st.subheader("Другие параметры")
    filters_ = copy.deepcopy(opts)
    cols = st.columns(4, gap='large')
    with cols[0]:
        date = st.date_input("Предполагемая дата сделки", pd.to_datetime("2024-05-31"))
        filters_.update({"instance_date": (pd.Timestamp('today') - pd.to_datetime(date)).days})
    with cols[1]:
        rms_vals = sorted(set(full_dataset["rooms_en"].values))
        rooms = st.selectbox("Количество комнат/тип комнаты", rms_vals, index=0)
        filters_.update({"rooms_en": rooms})
    with cols[2]:
        sq = st.number_input("Площадь квартиры, м2")
        filters_.update({"procedure_area": sq})
    with cols[3]:
        reg_type_en_vals = list(set(full_dataset["reg_type_en"].values))
        reg_type_en = st.selectbox("Сдан ли объект", reg_type_en_vals, index=1)
        filters_.update({"reg_type_en": reg_type_en})
        
    parking = st.checkbox("Есть ли паркинг", value=True)
    filters_.update({"has_parking": 1 if parking else 0})
    
    # print("filters_:", filters_)
    
    ############### Отображение полученного датасета ###############
    filters_handsome = {}
    for key in filters_:
        if filters_[key]:
            if isinstance(filters_[key], set):
                filters_handsome.update({key: list(filters_[key])[0]})
            else:
                filters_handsome.update({key: filters_[key]})
    # чтобы не выпадал паркинг при пересборке словаря
    filters_handsome.update({"has_parking": filters_["has_parking"]})
    
    print("All filters: ", filters_handsome)
    
    dataset = copy.deepcopy(full_dataset).reset_index(drop=True)
    dataset["actual_worth"] = dataset["actual_worth"].astype(int)
    for filter_key in filters_handsome:
        if (filter_key == "instance_date") or filters_handsome[filter_key] == 'no_data':
            continue
        dataset = dataset[dataset[filter_key] == filters_handsome[filter_key]]
    if len(dataset) > 0:
        st.dataframe(dataset.head(10).style.highlight_max(axis=0))
        
    ############### Работа с моделью ###############
    # - 1 - за таргетное значение
    if len(filters_handsome) == len(dataset.columns) - 1:
        cols_in_order = full_dataset.columns.to_list()[:-1]
        check_df = pd.DataFrame({i: [filters_handsome[i]] for i in filters_handsome})
        check_df = check_df[cols_in_order]
        value = round(model.predict(check_df)[0])
        st.write(f"Стоимость данной квартиры оценивается в: {value} дирхам")
    
    return opts
        

def main():
    all_combs = load_data()
    dataset = load_full_dataset()
    model = load_model()
    st.button("Сброс фильтров", on_click=reset_state_callback)
    
    tranformed_data = query_data(all_combs)
    print("Shape: ", tranformed_data[tranformed_data.selected == True].shape)
    
    current_query  = get_ui(cols, keys, tranformed_data, dataset, model)
    update_state(current_query)
    
    print("st.session_state.counter: ", st.session_state.counter)

st.set_page_config(layout="wide")
initialize_state() 
main()
    