import streamlit as st
import pandas as pd
import numpy as np
import math
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
import pydeck as pdk
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 데이터 로드 (공유)
# -------------------------------
@st.cache_data
def load_all():
    return (
        pd.read_csv('mid_student_master.csv'),
        pd.read_csv('high_student_master.csv'),
        pd.read_csv('brithrate_master.csv'),
        pd.read_csv('pop_master.csv'),
    )

mid_df, high_df, birthrate_df, pop_df = load_all()

# -------------------------------
# 공통 유틸 함수
# -------------------------------
def remove_outliers_iqr_per_school(school_data, min_samples=4):
    X = school_data['year'].values.reshape(-1,1)
    y = school_data['total_std'].values

    # 데이터 수가 적으면 이상치 제거 스킵
    if len(y) < min_samples:
       return X, y

    q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
    iqr = q3 - q1
    mask = (y >= q1 - 1.5*iqr) & (y <= q3 + 1.5*iqr)
    return X[mask], y[mask]


def get_birthrate_adjustment(gu_name, birth_year):
    try:
        row = birthrate_df[
            (birthrate_df['gu_name'] == gu_name) &
            (birthrate_df['birth_year'] == birth_year)
        ]
        return row['birthrate'].values[0]
    except:
        return 1.0


def train_and_predict_school(df, year, school_type):
    kernel = 1.0 * DotProduct(sigma_0=1.0) + WhiteKernel()
    results = []
    for name in df['school_name'].unique():
        sd = df[df['school_name'] == name]
        if sd.empty: continue
        br = get_birthrate_adjustment(
            sd.iloc[0]['gu_name'],
            year - (13 if school_type == 'mid' else 16)
        )
        Xc, yc = remove_outliers_iqr_per_school(sd)
        if len(np.unique(yc)) <= 2: continue
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(Xc, yc)
        ym, ys = model.predict([[year]], return_std=True)
        if ym[0] > 0:
            results.append({
                'school_name': name,
                'predicted_student_cnt': int(ym[0] * br),
                'prediction_lower': int((ym[0] - 1.96*ys[0]) * br)
            })
    return pd.DataFrame(results)


def compute_risk(pred_df, base_df, pop_decline, school_type):
    # 2024년 기준 학생 수 가져오기
    base_2024 = base_df[base_df['year'] == 2024][['school_name', 'total_std']].rename(columns={'total_std': 'student_cnt_2024'})
    
    # merge
    df = (
        pred_df[['school_name','predicted_student_cnt']]
        .merge(base_2024, on='school_name', how='left')
        .merge(base_df[['school_name','teacher_ratio_std']].drop_duplicates('school_name'), on='school_name', how='left')
        .dropna()
    )

    # 타입 변환
    df['teacher_ratio_std'] = pd.to_numeric(df['teacher_ratio_std'], errors='coerce')
    df['student_cnt_2024'] = pd.to_numeric(df['student_cnt_2024'], errors='coerce')
    df.dropna(subset=['teacher_ratio_std', 'student_cnt_2024'], inplace=True)

    # 1. 진짜 학생 감소율 계산
    df['decline_rate'] = ((df['student_cnt_2024'] - df['predicted_student_cnt']) / df['student_cnt_2024']) * 100

    # 2. 정규화
    scaler = MinMaxScaler()
    df['d_n'] = scaler.fit_transform(df[['decline_rate']])
    df['t_n'] = 1 - scaler.fit_transform(df[['teacher_ratio_std']])
    p_n = (pop_decline - (-10)) / 20

    # 3. 위험도 계산
    df['risk_score'] = df['d_n']*0.6 + df['t_n']*0.3 + p_n*0.1

    # 4. 평균 위험도 계산(대비)
    df = df.groupby('school_name', as_index=False)['risk_score'].mean()

    # 고등학교면 school_type 정보 추가
    if school_type == '고등학교':
        df = df.merge(
            base_df[['school_name','school_type']].drop_duplicates('school_name'),
            on='school_name', how='left'
        )

    return df.sort_values('risk_score', ascending=False).reset_index(drop=True)


def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# -------------------------------
# 1) 학교 재배치 위험도 지도 앱
# -------------------------------
def app_relocation():
    st.header("👩‍🎓 학교 재배치 위험도 분석 지도")
    school_kind = st.selectbox("학교 종류", ["중학교","고등학교"], key="reloc_kind")
    year = st.number_input("예측 연도(2025 ~ 2035)", 2025, 2035, 2025, key="reloc_year")
    # 고등학교 유형 선택
    if school_kind == '고등학교':
        types = high_df['school_type'].drop_duplicates().tolist()
        default = [t for t in types if '일반' in t]
        school_types = st.multiselect("포함할 고등학교 유형", types, default, key="reloc_types")
    else:
        school_types = None
    st.markdown("**🔵**: 위험도 0.50 미만.")
    st.markdown("**🟢**: 위험도 0.50 이상 0.60 미만.")
    st.markdown("**🟠**: 위험도 0.60 이상 0.70 미만.")
    st.markdown("**🔴**: 위험도 0.70 이상 0.80 미만.")
    st.markdown("**⚫**: 위험도 0.80 이상.")
    st.markdown("**위험도** = 학생 수 감소율 정규화 값 X 0.6 + 교사 수 대비 학생 수 비율 정규화 값 X 0.3 + 인구 감소율 정규화 값 X 0.1 ")
    if st.button("지도 생성", key="reloc_btn"):
        # 1) 예측 & 위험도 계산
        if school_kind == "중학교":
            pred     = train_and_predict_school(mid_df,  year, 'mid')
            pop_now  = pop_df.loc[pop_df['year']==year,   'mid'].iloc[0]
            pop_prev = pop_df.loc[pop_df['year']==year-1, 'mid'].iloc[0]
            base_df  = mid_df

        else:
            pred     = train_and_predict_school(high_df, year, 'high')
            pop_now  = pop_df.loc[pop_df['year']==year,   'high'].iloc[0]
            pop_prev = pop_df.loc[pop_df['year']==year-1, 'high'].iloc[0]
            base_df  = high_df

        decline = ((pop_now - pop_prev) / pop_prev) * 100
        risk_df = compute_risk(pred, base_df, decline, school_kind)
        # 선택한 고등학교 유형만 필터
        if school_types:
            risk_df = risk_df[risk_df['school_type'].isin(school_types)].reset_index(drop=True)
        # 3) 상위 10개 & 위치 merge
        latlon = base_df[['school_name','lat','lon']].drop_duplicates('school_name')
        merged = (
            risk_df
            .merge(pred[['school_name','predicted_student_cnt']], on='school_name')
            .merge(latlon, on='school_name', how='left')
            .dropna(subset=['lat','lon'])
            .head(10)
            .reset_index(drop=True)
        )

        # 4) 순위·전체학생수·색상 계산
        merged['순위']               = merged.index + 1
        merged['예측 학생수(전학년)'] = merged['predicted_student_cnt'] 
        def pick_color(r):
            if r >= 0.8:      return [0,   0,   0]   # 검정색
            elif r >= 0.7:    return [255, 0,   0]   # 빨간색
            elif r >= 0.6:    return [255, 165, 0]   # 주황색
            elif r >= 0.5:    return [0,   255, 0]   # 녹색
            else:             return [0,   0,   255] # 파란색
        merged['color'] = merged['risk_score'].apply(pick_color)

        # 5) pydeck 시각화
        center = [merged['lat'].mean(), merged['lon'].mean()]
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=merged,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=500,
            pickable=True
        )
        view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11)
        tooltip = {
            "html": "<b>{순위}. {school_name}</b><br>"
                    "예측 학생수(전학년): {예측 학생수(전학년)}<br>"
                    "위험도: {risk_score}",
            "style": {"color": "white"}
        }
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/streets-v11"
        ))

        # 6) 표 출력 (컬럼명 한글화)
        display_df = merged.rename(columns={
            'school_name': '학교명',
            'risk_score':  '위험도'
        })[
            ['순위','학교명','예측 학생수(전학년)','위험도']
        ].set_index('순위')

        st.dataframe(display_df)

# -------------------------------
# 2) 학교 합병 솔루션 앱
# -------------------------------
def app_merge():
    st.header("👩‍🎓 AI 기반 미래 학교 재배치 최적화 솔루션")
    # Menu
    school_kind = st.selectbox("학교 종류", ["중학교","고등학교"], key="merge_kind")
    year = st.number_input("예측 연도(2025 ~ 2035)", 2025, 2035, 2025, key="merge_year")
    max_pairs = st.slider("최대 학교 쌍 개수", 1, 5, 3, key="merge_pairs")
    include_girls = st.checkbox("여자학교 포함", True, key="merge_include")
    if school_kind == '고등학교':
        types = high_df['school_type'].drop_duplicates().tolist()
        default = [t for t in types if '일반' in t]
        school_types = st.multiselect("포함할 고등학교 유형", types, default, key="hs_types")
    else:
        school_types = None
    st.markdown("**방법**: 학교 용량(과거 최대 학생 수) & 통합하는 학교 간 거리 고려하여 고위험도 10개 학교 합병 가능성을 분석.")
    st.markdown("**전체 학생 수 차이 비율**: 두 학교 최대 학생 수 차이를 비율로 보여줍니다. 0에 가까울수록 비슷한 규모의 학교입니다.")
    if st.button("솔루션 생성", key="merge_btn"):
        # 데이터 준비
        df_src = mid_df if school_kind=='중학교' else high_df
        # capacity: 2024년 전체 학생 수
        cap_df = (
            df_src[df_src['year'] == 2024]
            .groupby('school_name', as_index=False)['student_cnt']
            .sum()
            .rename(columns={'student_cnt': 'capacity'})
        )
        pred = train_and_predict_school(df_src, year, 'mid' if school_kind=='중학교' else 'high')
        pop_now = pop_df.loc[pop_df['year']==year, 'mid' if school_kind=='중학교' else 'high'].iloc[0]
        pop_prev = pop_df.loc[pop_df['year']==year-1, 'mid' if school_kind=='중학교' else 'high'].iloc[0]
        risk_df = compute_risk(pred, df_src, (pop_now-pop_prev)/pop_prev*100, school_kind)
        if school_types:
            risk_df = risk_df[risk_df['school_type'].isin(school_types)]
        if not include_girls:
            risk_df = risk_df[~risk_df['school_name'].str.contains('여자')]
        # 2024년에 존재하는 학교만 남기기
        valid_schools = df_src[df_src['year'] == 2024]['school_name'].unique()
        risk_df = risk_df[risk_df['school_name'].isin(valid_schools)]
        # top10
        top10 = (
            risk_df.head(10)
            .merge(pred[['school_name','predicted_student_cnt']], on='school_name')
            .merge(cap_df, on='school_name', how='left')
            .merge(df_src[['school_name','lat','lon','teacher_ratio_std']].drop_duplicates('school_name'), on='school_name')
        )
        top10['순위']=top10.index+1
        
        # 후보 생성 (중복 없이)
        pairs = []
        for i,j in combinations(top10.index,2):
            r1,r2 = top10.loc[i], top10.loc[j]
            if ('여자' in r1.school_name) != ('여자' in r2.school_name): continue
            d = haversine(r1.lon,r1.lat,r2.lon,r2.lat)
            cd = abs(r1.capacity-r2.capacity)/max(r1.capacity,r2.capacity)
            pairs.append({'A':r1.school_name,'B':r2.school_name,'거리':d,'전체 학생 수 차이 비율':cd})
        cand_df = pd.DataFrame(pairs).sort_values(['거리','전체 학생 수 차이 비율'])
        selected = []
        used = set()
        for _, row in cand_df.iterrows():
            a, b = row['A'], row['B']
            if a not in used and b not in used:
                selected.append(row)
                used.update([a, b])
            if len(selected) >= max_pairs:
                break
        sol_df = pd.DataFrame(selected)
        
        merged = set(sol_df['A'])|set(sol_df['B'])
        top10['합병여부']=top10['school_name'].apply(lambda x:'✅합병' if x in merged else '❌미합병')
        top10['설명']=top10['school_name'].apply(lambda x: next((
            f"{r.A} ⇄ {r.B} (거리 {r.거리:.1f}km, 차이 {r['전체 학생 수 차이 비율']:.2f})" 
            for _,r in sol_df.iterrows() if x in [r.A,r.B]
        ), '❌ 통합 가능한 학교가 없습니다!'))
        # map
        color_map = {}
        for _, row in sol_df.iterrows():
            c = (np.random.rand(3) * 255).astype(int).tolist()
            color_map[row['A']] = c
            color_map[row['B']] = c
        top10['marker_color'] = top10['school_name'].apply(
            lambda x: color_map.get(x, [255, 0, 0])
        )

        data=[{k:(v.item() if isinstance(v,np.generic) else v) for k,v in row.items()} for row in top10.to_dict('records')]
        tooltip={"html":"<b>{school_name}</b><br>예측: {predicted_student_cnt}명<br>용량: {capacity}명<br>위험도: {risk_score}<br>{설명}","style":{"color":"white"}}
        layer=pdk.Layer("ScatterplotLayer",data=data,get_position=["lon","lat"],get_fill_color="marker_color",pickable=True,get_radius=500)
        view_state={"latitude":float(top10.lat.mean()),"longitude":float(top10.lon.mean()),"zoom":11}
        deck=pdk.Deck(layers=[layer],initial_view_state=view_state,tooltip=tooltip,map_style="mapbox://styles/mapbox/streets-v11")
        st.pydeck_chart(deck)
        # 테이블
        st.subheader('📊 위험도 상위 10개 학교')
        st.dataframe(
            top10
            .rename(columns={
                'school_name': '학교명',
                'predicted_student_cnt': '예측 전체 학생 수',
                'capacity': '2024년 기준 전체 학생 수',
                'risk_score': '위험도'
            })[[
                '순위',
                '학교명',
                '예측 전체 학생 수',
                '2024년 기준 전체 학생 수',
                '위험도',
                '합병여부'
            ]]
        )
        st.subheader('🔗 제안된 학교 쌍')

        st.dataframe(sol_df[['A','B','거리','전체 학생 수 차이 비율']].reset_index(drop=True))

# -------------------------------
# 메인: 탭으로 분리
# -------------------------------
def main():
    st.title("🎓 학교 분석 대시보드")
    tabs = st.tabs(["재배치 위험도 분석 지도", "AI 기반 학교 재배치 솔루션"])
    with tabs[0]:
        app_relocation()
    with tabs[1]:
        app_merge()


if __name__ == "__main__":
    main()

