#app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import japanize_matplotlib
import plotly.graph_objects as go

# with open(r'C:\Users\kazi1\OneDrive\デスクトップ\data_analysis\study_python\markdownfile\test',mode='r', encoding='utf-8') as f:
#     input_text = f.read()

# st.title('Hello,Stremlit')
# st.write('これは最初のStremlitアプリです')

# st.header('セクション1')
# st.markdown(input_text)
# st.code("print('Hello')",language='python')

# プルダウンメニューの作成やボタン、入力フォームの作成方法について
st.header('セクション2')
st.write('名前を教えて下さい')

name = st.text_input("名前を入力してください")

if st.button("送信"):
    st.write(f"こんにちは、{name}さん")
    
if st.checkbox("メッセージを表示する"):
    st.write('チェックしましたね')
    
option_list = ["リンゴ","バナナ","みかん"]
option = st.select_slider(
    "好きな果物を選んでください：",option_list
)

st.write("あなたが選んだ果物：", option)

st.header('セクション３')
st.write('カラムでレイアウト')

col1, col2 = st.columns(2)
# withを使うとカラムに含ませる要素をまとめて記載することができる。
with col1:
    st.header('左カラム')
    st.button('左ボタン')
with col2:
    st.header('右カラム')
    st.button('右ボタン')

# サイドバーの使い方
st.title("サイドバーの使い方")

name2 = st.sidebar.text_input('名前を入力してください。')
option2 = st.sidebar.selectbox("好きな色",["赤","青","緑"])


st.write(f"{name2} さんが選んだ色は {option2} です。")

col1, col2, col3 = st.columns([2, 1, 2])  # 幅の比率を調整

with col1:
    st.write("📊 グラフエリア（予定）")

with col2:
    st.write("🔘 操作ボタン（予定）")

with col3:
    st.write("📋 結果の要約（予定）")


st.title('簡単グラフ描画')


# ダミーデータを作成
df = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["A", "B", "C"]
)

st.subheader("折れ線グラフ")
st.line_chart(df)

st.subheader("棒グラフ")
st.bar_chart(df)

# st.title("matplotlibで描画")

# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title("サイン波")

# st.pyplot(fig)

# st.title("seabornで描画")
# # データセット読み込み（seabornのサンプル）
# df = sns.load_dataset("tips")

# fig, ax = plt.subplots()
# sns.boxplot(x="day", y="total_bill", data=df, ax=ax)
# st.pyplot(fig)

import plotly.express as px

st.title("Plotlyで動的なグラフ")

df = sns.load_dataset("iris")  # 有名な花のデータセット

fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    size="petal_length",
    title="Iris データの散布図"
)

st.plotly_chart(fig, use_container_width=True)

st.title("ヒートマップ（ランダムデータ）")

# ランダムなデータを作成
np.random.seed(0)
data = np.random.rand(10, 10)
df = pd.DataFrame(data, columns=[f"Col{i}" for i in range(10)])

fig = px.imshow(df, text_auto=True, color_continuous_scale='Viridis')
st.plotly_chart(fig, use_container_width=True)

st.title("カテゴリ別相関ヒートマップ")

# データの読み込み
df = sns.load_dataset("iris")

# 選択肢の作成
species_list = df["species"].unique()
selected_species = st.selectbox("品種を選んでください", species_list)

# 選んだカテゴリでフィルタ
filtered_df = df[df["species"] == selected_species]

# 数値列だけ抽出して相関行列を計算
numeric_df = filtered_df.select_dtypes(include="number")
corr = numeric_df.corr()

# ヒートマップ描画
fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    title=f"{selected_species} の相関ヒートマップ"
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(df.head())


st.title('Day6')
# データ読み込み
st.title("ペンギン可視化アプリ 🐧")
df = sns.load_dataset("penguins").dropna()  # 欠損を除外


# カテゴリでフィルタ
species = st.sidebar.multiselect("表示する種別", df["species"].unique(), default=list(df["species"].unique()))
filtered_df = df[df["species"].isin(species)]

# 数値列から軸を選ぶ
numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
x_axis = st.selectbox("X軸を選択", numeric_cols)
y_axis = st.selectbox("Y軸を選択", numeric_cols, index=1)

# グラフタイプを選択
chart_type = st.radio("表示グラフタイプ", ["散布図", "箱ひげ図"])

# グラフ表示
if chart_type == "散布図":
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color="species", title="散布図")
else:
    fig = px.box(filtered_df, x="species", y=y_axis, title="箱ひげ図")

st.plotly_chart(fig, use_container_width=True)

# 表示データも下に出力
with st.expander("📋 データを表示"):
    st.dataframe(filtered_df)

# 商品コードはstreamlit上で選択可能な形式として各商品ごとの前年度との売上の比較をするダッシュボードの作成
st.title('データの出力訓練')

upload_file = st.file_uploader('csvファイルをとりこんでください', type='csv')
if upload_file is not None:
    df = pd.read_csv(upload_file)
    df['category1'] = df['category'].str.split('|')
    df['category2'] = df['category1'].apply(lambda x:x[0] if x else None)
    df['discounted_price'] = df['discounted_price'].str.replace('₹','').str.replace(',','').astype('float')

    vizdata = df.groupby('category2').sum().reset_index()
    vizdata = vizdata[['category2','discounted_price']].sort_values('discounted_price',ascending=False)

    fig = px.bar(vizdata,x='category2',y='discounted_price')
    st.plotly_chart(fig, use_container_width=True)

def create_dataset():
    # 商品リストと期間
    products = ["商品A", "商品B", "商品C", "商品D"]
    months = pd.date_range(start="2024-01-01", end="2025-04-01", freq="MS")  # 月初日

    # データ生成
    np.random.seed(42)  # 再現性確保
    data = []

    for product in products:
        base_sales = np.random.randint(80, 150) * 1000
        seasonal_fluctuation = np.sin(np.linspace(0, 3 * np.pi, len(months))) * 20000  # 季節性
        random_noise = np.random.normal(0, 10000, len(months))  # ノイズ

        sales = base_sales + seasonal_fluctuation + random_noise
        sales = np.maximum(sales, 0).astype(int)  # 売上が負にならないように調整

        for date, amount in zip(months, sales):
            data.append({
                "年月": date.strftime("%Y-%m"),
                "商品名": product,
                "売上金額": amount
            })

    df = pd.DataFrame(data)
    return df

sales_data = create_dataset()
product = st.selectbox(label="商品名を選んでください。",options=sales_data['商品名'].unique())

viz_data = sales_data.groupby(['商品名','年月']).sum().reset_index()
viz_data[['year','month']]=viz_data['年月'].str.split('-', expand=True)

viz_data_2024_p = viz_data.loc[(viz_data['year']=='2024')*
                              (viz_data['商品名']==product)]
viz_data_2025_p = viz_data.loc[(viz_data['year']=='2025')*
                              (viz_data['商品名']==product)]

# カスタムグラフの作成
fig = go.Figure()

# 折れ線：2024年
fig.add_trace(go.Scatter(
    x=viz_data_2024_p['month'],
    y=viz_data_2024_p['売上金額'],
    mode='lines+markers',
    name='2024年（折れ線）'
))

# 棒グラフ：2025年
fig.add_trace(go.Bar(
    x=viz_data_2025_p['month'],
    y=viz_data_2025_p['売上金額'],
    name='2025年（棒グラフ）'
))

# レイアウト調整
fig.update_layout(
    title='商品Aの年度別売上（2024年: 折れ線、2025年: 棒グラフ）',
    xaxis_title='month',
    yaxis_title='売上金額',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 データを表示"):
    st.dataframe(sales_data)