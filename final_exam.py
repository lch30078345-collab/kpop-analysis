import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import altair as alt
import plotly.express as px
import networkx as nx
import urllib.request
import json
import re
from konlpy.tag import Okt
from wordcloud import WordCloud
from collections import Counter
from itertools import combinations

# 1. 페이지 및 폰트 
st.set_page_config(page_title="K팝 데몬 헌터스 분석", layout="wide")


font_path = "Pretendard-Regular.ttf"
stopwords_path = "stopwords.txt"

fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# 2. 학번 및 이름 
st.title("K팝 데몬 헌터스: 팬덤 형성의 핵심 요인 분석")
st.subheader("학번 : C321058  |  이름 : 이채희")
st.markdown("""
> **기획 의도:** 본 대시보드는 2025년 화제작 'K팝 데몬 헌터스'에 대한 온라인 여론을 실시간으로 수집하여 분석합니다.  
> 단순한 언급량을 넘어서, 대중이 어떤 키워드에 반응하고 있으며, 키워드 간에 어떤 연결성이 있는지 파악하여 팬덤 형성의 원동력을 도출하는 것을 목표로 합니다.
""")
st.write("---")

# 3. 사이드바 위젯 구성
st.sidebar.header("Step 1. 데이터 수집 설정")
query = st.sidebar.text_input("검색어 입력", "K팝 데몬 헌터스")
display_count = st.sidebar.slider("수집할 기사 수", 10, 100, 100, 10)
sort_option = st.sidebar.selectbox("정렬 기준", ["sim", "date"])
collect_btn = st.sidebar.button("데이터 수집 및 분석 시작")

st.sidebar.divider()
st.sidebar.header("Step 2. 시각화 옵션")
wc_bg = st.sidebar.radio("워드클라우드 배경", ["white", "black"], horizontal=True)
min_edge = st.sidebar.slider("네트워크 연결 최소 빈도", 1, 15, 3)

# 4. 데이터 수집 
if collect_btn:
    client_id = "BAa7WmdQwBpItekevUgc"
    client_secret = "BRRmddBdNS"
    
    encText = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/news.json?query={encText}&display={display_count}&sort={sort_option}"
    
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    
    response = urllib.request.urlopen(request)
    if response.getcode() == 200:
        data = json.loads(response.read().decode('utf-8'))
        df = pd.DataFrame(data['items'])
        
        df['pubDate'] = pd.to_datetime(df['pubDate']).dt.date
        df['title'] = df['title'].str.replace('<b>', '').str.replace('</b>', '').str.replace('&quot;', '')
        df['description'] = df['description'].str.replace('<b>', '').str.replace('</b>', '').str.replace('&quot;', '')
        
        df.to_csv('collected_data.csv', index=False, encoding='utf-8-sig')
        st.session_state.df = df

# 5. 분석 및 시각화
if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df
    
    # [데이터 확인]
    st.subheader("1. 데이터 수집 현황")
    st.dataframe(df.head())
    st.info(f"**Data Insight:** 총 {len(df)}건의 최신 기사가 수집되었습니다. 핵심 키워드를 추출합니다.")

    okt = Okt()
    
    # 불용어 처리 (파일 + 직접 추가)
    stopwords = []
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = f.read().splitlines()
    except:
        pass 

    # 뉴스 기사 잡음 제거용 불용어 추가
    extra_stopwords = ['뉴스', '기사', '기자', '지난', '위해', '통해', '관련', '대한', '경우', '가장', '이번', '때문', '정도', '대해', '무단', '배포', '금지', '전재', '속보', '오늘', '등', '및', '바','스케','제이지','매기']
    stopwords.extend(extra_stopwords)
    stopwords = list(set(stopwords)) # 중복 제거
    
    all_nouns = []
    sentences = []
    text_data = df['title'] + " " + df['description']
    
    for text in text_data:
        clean_text = re.sub("[^가-힣 ]", "", text)
        nouns = [n for n in okt.nouns(clean_text) if len(n) > 1 and n not in stopwords]
        all_nouns.extend(nouns)
        sentences.append(nouns)

    count = Counter(all_nouns)
    top_20 = pd.DataFrame(count.most_common(20), columns=['단어', '빈도'])

    st.write("---")
    
    # Seaborn
    st.subheader("2. 핵심 이슈 키워드 (Seaborn)")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_20, x='빈도', y='단어', ax=ax1, palette='viridis')
    st.pyplot(fig1)
    st.markdown("""
    **👉 해석:** 빈도수가 높은 상위 키워드들은 현재 **대중이 가장 주목하는 요소**입니다. 
    특정 멤버의 이름이나 곡명, '넷플릭스' 등의 플랫폼 이름이 상위에 있다면 그것이 팬덤 유입의 주 경로임을 시사합니다.
    """)

    st.write("---")

    # Plotly
    st.subheader("3. 키워드 점유율 분석 (Plotly)")
    fig2 = px.pie(top_20.head(10), values='빈도', names='단어', hole=0.3, title="상위 10개 키워드 비중")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    **👉 해석:** 상위 10개 키워드가 전체 이슈에서 차지하는 비중입니다. 
    특정 키워드의 비중이 압도적이라면, 팬덤의 관심사가 **하나의 이슈에 집중**되어 있음을 의미합니다.
    """)

    st.write("---")

    # Altair
    st.subheader("4. 시계열 트렌드 변화 (Altair)")
    trend_df = df['pubDate'].value_counts().reset_index()
    trend_df.columns = ['날짜', '기사수']
    trend_df = trend_df.sort_values('날짜')
    
    chart = alt.Chart(trend_df).mark_line(point=True, color='red').encode(
        x='날짜', y='기사수', tooltip=['날짜', '기사수']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.info("👉 **Trend Insight:** 그래프가 급격히 상승하는 시점에 주요 이벤트(티저, 발매 등)가 있었는지 파악할 수 있습니다.")

    st.write("---")

    # WordCloud
    st.subheader("5. 종합 이슈 워드클라우드")
    
    wc = WordCloud(
        font_path=font_path, 
        background_color='white',  
        width=900, 
        height=500, 
        colormap='cool',  
        max_words=30
    )
    gen = wc.generate_from_frequencies(dict(count.most_common(30)))
    
    fig3 = plt.figure(figsize=(12, 6))
    plt.imshow(gen)
    plt.axis('off')
    st.pyplot(fig3)
    st.markdown("**👉 요약:** 텍스트 크기가 클수록 팬덤 내에서 언급된 횟수가 많은 의미 있는 단어입니다.")
    st.write("---")

    # NetworkX
    st.subheader("6. 키워드 동시출현 네트워크")
    st.caption(f" 현재 설정된 최소 빈도: {min_edge} (사이드바에서 조절 가능)")
    
    edges = []
    for s in sentences:
        for a, b in combinations(s, 2):
            edges.append(tuple(sorted((a, b))))
            
    edge_counts = Counter(edges)
    final_edges = [(a, b, c) for (a, b), c in edge_counts.items() if c >= min_edge]

    if final_edges:
        G = nx.Graph()
        G.add_weighted_edges_from(final_edges)
        
        
        fig4, ax4 = plt.subplots(figsize=(15, 15))
        
        # k=1.5로 노드 간격을 넓힘
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        d = dict(G.degree)
        node_size = [v * 120 for v in d.values()] # 노드 크기 확대
        
        # 노드
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.9)
        
        # 엣지 
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=1.0)
        
        # 라벨 
        nx.draw_networkx_labels(G, pos, font_family=font_name, font_size=12, font_weight='bold')
        
        plt.axis('off')
        st.pyplot(fig4)
        
        st.success("""
        **👉 네트워크 시각화 해석:**
        * **노드 (Node, 점):** 추출된 핵심 명사입니다. 점이 클수록 많은 단어와 연결된 **'중심(Degree Centrality)'** 키워드입니다.
        * **엣지 (Edge, 선):** 두 단어가 같은 문장에서 함께 등장한 **'동시 출현(Co-occurrence)'** 관계를 나타냅니다.
        * **결론:** 네트워크 중앙에 밀집되어 서로 복잡하게 연결된 단어들이 이번 이슈를 관통하는 핵심 주제입니다.
        """)
    else:
        st.warning("연결된 키워드가 없습니다. 사이드바에서 '최소 빈도'를 낮춰보세요.")

else:
    st.info("👈 왼쪽 사이드바에서 '데이터 수집 및 분석 시작' 버튼을 눌러주세요.")