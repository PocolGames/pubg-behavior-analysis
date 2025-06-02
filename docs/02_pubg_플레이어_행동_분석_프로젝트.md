# PUBG 플레이어 행동 분석 프로젝트

## Phase 1: 환경 설정 및 데이터 준비

``` python
import warnings
warnings.filterwarnings('ignore')
```

### 1. 필수 라이브러리 설치 및 임포트


``` python
# 1. 필수 라이브러리 설치 및 임포트

# 기본 데이터 처리 라이브러리
import pandas as pd
import numpy as np
import os
import json
import gc
from pathlib import Path

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 딥러닝 라이브러리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 통계 및 기타 라이브러리
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
import datetime
import time

print("✅ 필수 라이브러리 로딩 완료!")
```

### 2. Google Colab 환경 최적화
``` python
# 2. Google Colab 환경 최적화

# GPU 사용 가능 여부 확인
print("🔧 시스템 환경 확인:")
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {tf.config.list_physical_devices('GPU')}")

# 메모리 사용량 확인 함수
def check_memory_usage():
    """현재 메모리 사용량을 확인하는 함수"""
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 메모리 사용량: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")

# 플롯 설정 최적화
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Plotly 설정
pyo.init_notebook_mode(connected=True)

print("✅ 환경 최적화 완료!")
```
#### 2. Google Colab 환경 최적화 결과

``` bash
🔧 시스템 환경 확인:
TensorFlow 버전: 2.18.0
GPU 사용 가능: []
✅ 환경 최적화 완료!
```

### 3. Kaggle API 설정 및 데이터 다운로드
``` python
# 3. Kaggle API 설정 및 데이터 다운로드

def setup_kaggle_api():
    """Kaggle API 설정 함수"""
    try:
        # Colab에서 kaggle.json 파일 업로드 확인
        from google.colab import files

        print("📁 Kaggle API 키 설정:")
        print("1. Kaggle 계정 > Account > API > Create New API Token")
        print("2. 다운로드된 kaggle.json 파일을 업로드하세요")

        # 파일 업로드 인터페이스 제공
        uploaded = files.upload()

        # kaggle.json 파일 이동 및 권한 설정
        os.makedirs('/root/.kaggle', exist_ok=True)
        os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
        os.chmod('/root/.kaggle/kaggle.json', 600)

        print("✅ Kaggle API 설정 완료!")
        return True

    except Exception as e:
        print(f"❌ Kaggle API 설정 실패: {e}")
        return False

def download_pubg_data():
    """PUBG 데이터셋 다운로드 함수"""
    try:
        # 데이터 디렉토리 생성
        os.makedirs('/content/pubg_data', exist_ok=True)
        os.chdir('/content/pubg_data')

        # Kaggle에서 데이터셋 다운로드
        print("📥 PUBG 데이터셋 다운로드 중...")
        os.system('kaggle competitions download -c pubg-finish-placement-prediction')

        # 압축 해제
        print("📦 파일 압축 해제 중...")
        os.system('unzip -q pubg-finish-placement-prediction.zip')

        # 파일 목록 확인
        files = os.listdir('.')
        print("📂 다운로드된 파일 목록:")
        for file in files:
            size = os.path.getsize(file) / (1024**2)  # MB 단위
            print(f"  - {file}: {size:.1f}MB")

        print("✅ 데이터 다운로드 완료!")
        return True

    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        return False

# Kaggle API 설정 및 데이터 다운로드 실행
print("🚀 Kaggle 데이터 준비 시작:")
print("\n" + "="*50)
print("📌 수동 실행 필요:")
print("1. setup_kaggle_api() 함수를 실행하여 API 키 설정")
print("2. download_pubg_data() 함수를 실행하여 데이터 다운로드")
print("="*50)
```
#### Kagle API 설정 및 데이터 다운로드 결과

``` bash
🚀 Kaggle 데이터 준비 시작:

==================================================
📌 수동 실행 필요:
1. setup_kaggle_api() 함수를 실행하여 API 키 설정
2. download_pubg_data() 함수를 실행하여 데이터 다운로드
==================================================
```

### 4. 데이터 로딩 및 기본 탐색 함수
``` python
# 4. 데이터 로딩 및 기본 탐색 함수

def load_pubg_data(sample_size=None, random_state=42):
    """
    PUBG 데이터를 효율적으로 로딩하는 함수

    Parameters:
    - sample_size: 샘플링할 행 수 (None이면 전체 데이터)
    - random_state: 랜덤 시드
    """
    try:
        print("📊 PUBG 데이터 로딩 시작...")

        # 파일 경로 확인
        data_path = '/content/pubg_data/train_V2.csv'
        if not os.path.exists(data_path):
            print("❌ train_V2.csv 파일을 찾을 수 없습니다!")
            return None

        # 파일 크기 확인
        file_size = os.path.getsize(data_path) / (1024**2)
        print(f"📁 파일 크기: {file_size:.1f}MB")

        # 메모리 사용량 확인
        check_memory_usage()

        # 데이터 로딩 (샘플링 옵션)
        if sample_size:
            print(f"🎯 {sample_size:,}행 샘플링으로 로딩...")
            # 전체 행 수 확인
            total_lines = sum(1 for line in open(data_path)) - 1
            skip_rows = sorted(np.random.choice(range(1, total_lines + 1),
                                              total_lines - sample_size,
                                              replace=False))
            df = pd.read_csv(data_path, skiprows=skip_rows)
        else:
            print("📈 전체 데이터 로딩...")
            df = pd.read_csv(data_path)

        print(f"✅ 데이터 로딩 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
        return df

    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None

def explore_basic_data(df):
    """기본 데이터 탐색 함수"""
    if df is None:
        print("❌ 데이터가 없습니다!")
        return

    print("\n" + "="*50)
    print("📋 기본 데이터 정보")
    print("="*50)

    # 기본 정보
    print(f"📊 데이터 크기: {df.shape[0]:,}행 x {df.shape[1]}열")
    print(f"💾 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")

    # 컬럼 정보
    print(f"\n📝 컬럼 목록 ({len(df.columns)}개):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  {i:2d}. {col:<20} | {str(dtype):<10} | 결측치: {null_count:>6}개 ({null_pct:.1f}%)")

    # 기본 통계
    print(f"\n📈 기본 통계:")
    print(df.describe())

    # 타겟 변수 분포 (winPlacePerc)
    if 'winPlacePerc' in df.columns:
        print(f"\n🎯 타겟 변수 (winPlacePerc) 분포:")
        print(df['winPlacePerc'].describe())

    # 샘플 데이터
    print(f"\n🔍 샘플 데이터 (상위 3행):")
    print(df.head(3))

    return df

# 데이터 로딩 테스트 함수
def test_data_loading():
    """데이터 로딩 테스트"""
    print("🧪 데이터 로딩 테스트:")

    # 작은 샘플로 테스트
    test_df = load_pubg_data(sample_size=1000)
    if test_df is not None:
        explore_basic_data(test_df)
        print("✅ 데이터 로딩 테스트 성공!")
        return True
    else:
        print("❌ 데이터 로딩 테스트 실패!")
        return False
```
### 5. 프로젝트 상태 확인 함수
``` python
# 5. 프로젝트 상태 확인 함수

def check_project_status():
    """프로젝트 현재 상태 확인"""
    print("\n" + "🎮 PUBG 플레이어 행동 분석 프로젝트")
    print("="*60)
    print("📍 현재 단계: Phase 1 - 환경 설정 및 데이터 준비")
    print("="*60)

    # 체크리스트
    checklist = [
        ("필수 라이브러리 설치", True),
        ("환경 최적화 설정", True),
        ("Kaggle API 함수 준비", True),
        ("데이터 로딩 함수 준비", True),
        ("기본 탐색 함수 준비", True)
    ]

    print("✅ 완료된 작업:")
    for task, completed in checklist:
        status = "✅" if completed else "⏳"
        print(f"  {status} {task}")

    print("\n📋 다음 단계:")
    print("  1. setup_kaggle_api() 실행")
    print("  2. download_pubg_data() 실행")
    print("  3. test_data_loading() 실행")
    print("  4. Phase 2로 진행")

    print("\n💡 메모리 절약 팁:")
    print("  - 전체 데이터가 크면 sample_size=800000 사용")
    print("  - 필요시 gc.collect()로 메모리 정리")
    print("  - GPU 런타임 권장")

# 프로젝트 상태 확인
check_project_status()

print("\n🚀 Phase 1 준비 완료!")
print("📱 다음 단계를 실행하여 데이터를 준비하세요:")
print("   setup_kaggle_api()")
print("   download_pubg_data()")
print("   test_data_loading()")
```
#### 5. 프로젝트 상태 확인 함수 결과

``` bash
🎮 PUBG 플레이어 행동 분석 프로젝트
============================================================
📍 현재 단계: Phase 1 - 환경 설정 및 데이터 준비
============================================================
✅ 완료된 작업:
  ✅ 필수 라이브러리 설치
  ✅ 환경 최적화 설정
  ✅ Kaggle API 함수 준비
  ✅ 데이터 로딩 함수 준비
  ✅ 기본 탐색 함수 준비

📋 다음 단계:
  1. setup_kaggle_api() 실행
  2. download_pubg_data() 실행
  3. test_data_loading() 실행
  4. Phase 2로 진행

💡 메모리 절약 팁:
  - 전체 데이터가 크면 sample_size=800000 사용
  - 필요시 gc.collect()로 메모리 정리
  - GPU 런타임 권장

🚀 Phase 1 준비 완료!
📱 다음 단계를 실행하여 데이터를 준비하세요:
   setup_kaggle_api()
   download_pubg_data()
   test_data_loading()
```

### 실행
``` python
setup_kaggle_api()
download_pubg_data()
test_data_loading()
```
#### 실행 결과

``` bash
📁 Kaggle API 키 설정:
1. Kaggle 계정 > Account > API > Create New API Token
2. 다운로드된 kaggle.json 파일을 업로드하세요
kaggle.json
kaggle.json(application/json) - 67 bytes, last modified: 2025. 5. 26. - 100% done
Saving kaggle.json to kaggle.json
✅ Kaggle API 설정 완료!
📥 PUBG 데이터셋 다운로드 중...
📦 파일 압축 해제 중...
📂 다운로드된 파일 목록:
  - test_V2.csv: 260.6MB
  - train_V2.csv: 629.0MB
  - pubg-finish-placement-prediction.zip: 360.9MB
  - sample_submission_V2.csv: 31.4MB
✅ 데이터 다운로드 완료!
🧪 데이터 로딩 테스트:
📊 PUBG 데이터 로딩 시작...
📁 파일 크기: 629.0MB
💾 메모리 사용량: 15.0% (1.6GB / 12.7GB)
🎯 1,000행 샘플링으로 로딩...
✅ 데이터 로딩 완료: 1,000행 x 29열
```

``` bash
==================================================
📋 기본 데이터 정보
==================================================
📊 데이터 크기: 1,000행 x 29열
💾 메모리 사용량: 0.5MB

📝 컬럼 목록 (29개):
   1. Id                   | object     | 결측치:      0개 (0.0%)
   2. groupId              | object     | 결측치:      0개 (0.0%)
   3. matchId              | object     | 결측치:      0개 (0.0%)
   4. assists              | int64      | 결측치:      0개 (0.0%)
   5. boosts               | int64      | 결측치:      0개 (0.0%)
   6. damageDealt          | float64    | 결측치:      0개 (0.0%)
   7. DBNOs                | int64      | 결측치:      0개 (0.0%)
   8. headshotKills        | int64      | 결측치:      0개 (0.0%)
   9. heals                | int64      | 결측치:      0개 (0.0%)
  10. killPlace            | int64      | 결측치:      0개 (0.0%)
  11. killPoints           | int64      | 결측치:      0개 (0.0%)
  12. kills                | int64      | 결측치:      0개 (0.0%)
  13. killStreaks          | int64      | 결측치:      0개 (0.0%)
  14. longestKill          | float64    | 결측치:      0개 (0.0%)
  15. matchDuration        | int64      | 결측치:      0개 (0.0%)
  16. matchType            | object     | 결측치:      0개 (0.0%)
  17. maxPlace             | int64      | 결측치:      0개 (0.0%)
  18. numGroups            | int64      | 결측치:      0개 (0.0%)
  19. rankPoints           | int64      | 결측치:      0개 (0.0%)
  20. revives              | int64      | 결측치:      0개 (0.0%)
  21. rideDistance         | float64    | 결측치:      0개 (0.0%)
  22. roadKills            | int64      | 결측치:      0개 (0.0%)
  23. swimDistance         | float64    | 결측치:      0개 (0.0%)
  24. teamKills            | int64      | 결측치:      0개 (0.0%)
  25. vehicleDestroys      | int64      | 결측치:      0개 (0.0%)
  26. walkDistance         | float64    | 결측치:      0개 (0.0%)
  27. weaponsAcquired      | int64      | 결측치:      0개 (0.0%)
  28. winPoints            | int64      | 결측치:      0개 (0.0%)
  29. winPlacePerc         | float64    | 결측치:      0개 (0.0%)

📈 기본 통계:
           assists       boosts  damageDealt        DBNOs  headshotKills  \
count  1000.000000  1000.000000  1000.000000  1000.000000    1000.000000   
mean      0.221000     1.159000   130.682024     0.639000       0.214000   
std       0.574882     1.805818   162.853553     1.040077       0.523907   
min       0.000000     0.000000     0.000000     0.000000       0.000000   
25%       0.000000     0.000000     0.000000     0.000000       0.000000   
50%       0.000000     0.000000    89.145000     0.000000       0.000000   
75%       0.000000     2.000000   181.900000     1.000000       0.000000   
max       4.000000    11.000000  1168.000000     7.000000       4.000000   

             heals    killPlace   killPoints        kills  killStreaks  ...  \
count  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000  ...   
mean      1.467000    47.177000   472.241000     0.942000     0.541000  ...   
std       2.826818    27.719561   617.898381     1.516884     0.694839  ...   
min       0.000000     1.000000     0.000000     0.000000     0.000000  ...   
25%       0.000000    23.000000     0.000000     0.000000     0.000000  ...   
50%       0.000000    46.000000     0.000000     0.000000     0.000000  ...   
75%       2.000000    71.000000  1152.000000     1.000000     1.000000  ...   
max      25.000000   100.000000  1882.000000    12.000000     5.000000  ...   

           revives  rideDistance    roadKills  swimDistance    teamKills  \
count  1000.000000   1000.000000  1000.000000   1000.000000  1000.000000   
mean      0.153000    584.856341     0.002000      2.933933     0.027000   
std       0.442478   1411.430874     0.044699     16.956302     0.168224   
min       0.000000      0.000000     0.000000      0.000000     0.000000   
25%       0.000000      0.000000     0.000000      0.000000     0.000000   
50%       0.000000      0.000000     0.000000      0.000000     0.000000   
75%       0.000000     37.392500     0.000000      0.000000     0.000000   
max       5.000000  10730.000000     1.000000    238.300000     2.000000   

       vehicleDestroys  walkDistance  weaponsAcquired    winPoints  \
count      1000.000000   1000.000000       1000.00000  1000.000000   
mean          0.008000   1174.950077          3.60100   568.488000   
std           0.089129   1232.368923          2.33779   731.804729   
min           0.000000      0.000000          0.00000     0.000000   
25%           0.000000    158.550000          2.00000     0.000000   
50%           0.000000    682.600000          3.00000     0.000000   
75%           0.000000   1919.250000          5.00000  1492.000000   
max           1.000000   7383.000000         19.00000  1755.000000   

       winPlacePerc  
count   1000.000000  
mean       0.476836  
std        0.312325  
min        0.000000  
25%        0.192300  
50%        0.474050  
75%        0.750000  
max        1.000000  

[8 rows x 25 columns]

🎯 타겟 변수 (winPlacePerc) 분포:
count    1000.000000
mean        0.476836
std         0.312325
min         0.000000
25%         0.192300
50%         0.474050
75%         0.750000
max         1.000000
Name: winPlacePerc, dtype: float64

🔍 샘플 데이터 (상위 3행):
               Id         groupId         matchId  assists  boosts  \
0  1d4802d5196bfc  5f7f2fcef7ffd2  4ae569cd5839bd        0       0   
1  adfb57db50ff99  8be5becce02129  9b3251ccfd56be        0       0   
2  23df6be369873d  6ef4e4dbd06580  44d4f3c1d95b55        0       0   

   damageDealt  DBNOs  headshotKills  heals  killPlace  ...  revives  \
0         53.0      1              0      0         40  ...        0   
1          0.0      0              0      0         36  ...        0   
2          0.0      0              0      0         73  ...        0   

   rideDistance  roadKills  swimDistance  teamKills vehicleDestroys  \
0           0.0          0           0.0          0               0   
1           0.0          0           0.0          0               0   
2           0.0          0           0.0          0               0   

   walkDistance  weaponsAcquired  winPoints  winPlacePerc  
0         40.94                1          0        0.0741  
1       1614.00                6       1497        0.7083  
2         57.67                1          0        0.2308  

[3 rows x 29 columns]
✅ 데이터 로딩 테스트 성공!
True
```

## Phase 2: 데이터 품질 관리 및 전처리

### 1. 데이터 품질 검사 함수들
``` python
print("🔍 Phase 2: 데이터 품질 관리 시작!")
print("="*60)
# 1. 데이터 품질 검사 함수들

def comprehensive_data_quality_check(df):
    """종합적인 데이터 품질 검사"""
    print("📊 종합 데이터 품질 검사")
    print("-" * 40)

    quality_report = {}

    # 1. 기본 정보
    print(f"📈 데이터 크기: {df.shape[0]:,}행 x {df.shape[1]}열")
    quality_report['shape'] = df.shape

    # 2. 결측치 분석
    print(f"\n💥 결측치 분석:")
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100

    missing_summary = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Count', ascending=False)

    print(missing_summary[missing_summary['Missing_Count'] > 0])
    quality_report['missing_data'] = missing_summary

    # 3. 중복 데이터 확인
    duplicates = df.duplicated().sum()
    print(f"\n🔄 중복 행 수: {duplicates:,}개 ({duplicates/len(df)*100:.2f}%)")
    quality_report['duplicates'] = duplicates

    # 4. 데이터 타입 확인
    print(f"\n📝 데이터 타입 분포:")
    dtype_counts = df.dtypes.value_counts()
    print(dtype_counts)
    quality_report['dtypes'] = dtype_counts

    return quality_report

def check_game_logic_violations(df):
    """게임 로직 기반 데이터 검증"""
    print("\n🎮 게임 로직 기반 데이터 검증")
    print("-" * 40)

    violations = {}

    # 1. 음수값 확인 (음수가 나올 수 없는 컬럼들)
    non_negative_cols = ['kills', 'knockdowns', 'assists', 'revives', 'heals', 'boosts',
                        'weaponsAcquired', 'vehicleDestroys', 'roadKills', 'teamKills',
                        'DBNOs', 'walkDistance', 'rideDistance', 'swimDistance']

    for col in non_negative_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                violations[f'{col}_negative'] = negative_count
                print(f"❌ {col}: {negative_count:,}개의 음수값")

    # 2. 논리적 불일치 확인
    if 'kills' in df.columns and 'DBNOs' in df.columns:
        # 킬 수가 다운 수보다 많은 경우 (논리적으로 불가능)
        kill_dbno_violation = (df['kills'] > df['DBNOs']).sum()
        if kill_dbno_violation > 0:
            violations['kills_gt_dbnos'] = kill_dbno_violation
            print(f"❌ 킬 수 > 다운 수: {kill_dbno_violation:,}개")

    # 3. 극단값 확인
    if 'kills' in df.columns:
        max_kills = df['kills'].max()
        extreme_kills = (df['kills'] > 30).sum()
        print(f"⚠️  최대 킬 수: {max_kills}, 30킬 이상: {extreme_kills:,}개")
        violations['extreme_kills'] = extreme_kills

    # 4. winPlacePerc 범위 확인
    if 'winPlacePerc' in df.columns:
        invalid_winplace = ((df['winPlacePerc'] < 0) | (df['winPlacePerc'] > 1)).sum()
        if invalid_winplace > 0:
            violations['invalid_winplaceperc'] = invalid_winplace
            print(f"❌ winPlacePerc 범위 오류 (0-1 밖): {invalid_winplace:,}개")

    print(f"\n📊 총 {len(violations)}개의 데이터 품질 이슈 발견")
    return violations

def detect_outliers_statistical(df, method='iqr'):
    """통계적 이상치 탐지"""
    print(f"\n📈 통계적 이상치 탐지 ({method.upper()} 방법)")
    print("-" * 40)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}

    for col in numeric_cols:
        if col in ['Id', 'groupId', 'matchId']:  # ID 컬럼 제외
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z_scores > 3).sum()

        if outliers > 0:
            outliers_info[col] = {
                'count': outliers,
                'percentage': (outliers / len(df)) * 100
            }

    # 상위 10개 이상치 컬럼 출력
    sorted_outliers = sorted(outliers_info.items(),
                           key=lambda x: x[1]['count'],
                           reverse=True)[:10]

    for col, info in sorted_outliers:
        print(f"📊 {col:<20}: {info['count']:>6,}개 ({info['percentage']:>5.1f}%)")

    return outliers_info
```

### 2. 데이터 정제 함수들
``` python
# 2. 데이터 정제 함수들

def clean_missing_values(df, strategy='smart'):
    """결측치 처리"""
    print(f"\n🧹 결측치 처리 (전략: {strategy})")
    print("-" * 40)

    df_cleaned = df.copy()

    if strategy == 'smart':
        # 스마트 전략: 컬럼별 맞춤 처리

        # 1. winPlacePerc 결측치 제거 (타겟 변수)
        if 'winPlacePerc' in df_cleaned.columns:
            before_count = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=['winPlacePerc'])
            removed = before_count - len(df_cleaned)
            print(f"📍 winPlacePerc 결측치 제거: {removed:,}행")

        # 2. 수치형 변수는 0으로 대체 (게임에서 행동하지 않은 것으로 해석)
        numeric_fill_zero = ['kills', 'knockdowns', 'assists', 'revives', 'heals', 'boosts',
                           'weaponsAcquired', 'vehicleDestroys', 'roadKills', 'teamKills', 'DBNOs']

        for col in numeric_fill_zero:
            if col in df_cleaned.columns:
                filled = df_cleaned[col].isnull().sum()
                df_cleaned[col] = df_cleaned[col].fillna(0)
                if filled > 0:
                    print(f"🔧 {col}: {filled:,}개를 0으로 대체")

        # 3. 거리 관련 변수도 0으로 대체
        distance_cols = ['walkDistance', 'rideDistance', 'swimDistance']
        for col in distance_cols:
            if col in df_cleaned.columns:
                filled = df_cleaned[col].isnull().sum()
                df_cleaned[col] = df_cleaned[col].fillna(0)
                if filled > 0:
                    print(f"🔧 {col}: {filled:,}개를 0으로 대체")

    elif strategy == 'drop':
        # 결측치가 있는 행 모두 제거
        before_count = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        removed = before_count - len(df_cleaned)
        print(f"🗑️ 결측치 포함 행 제거: {removed:,}행")

    print(f"✅ 정제 후 데이터 크기: {df_cleaned.shape[0]:,}행")
    return df_cleaned

def handle_duplicates(df, strategy='remove'):
    """중복 데이터 처리"""
    print(f"\n🔄 중복 데이터 처리 (전략: {strategy})")
    print("-" * 40)

    df_cleaned = df.copy()

    if strategy == 'remove':
        before_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed = before_count - len(df_cleaned)
        print(f"🗑️ 중복 행 제거: {removed:,}행")

    elif strategy == 'keep_first':
        before_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(keep='first')
        removed = before_count - len(df_cleaned)
        print(f"🥇 중복 행 중 첫 번째만 유지: {removed:,}행 제거")

    print(f"✅ 처리 후 데이터 크기: {df_cleaned.shape[0]:,}행")
    return df_cleaned

def fix_game_logic_violations(df):
    """게임 로직 위반 수정"""
    print(f"\n🎮 게임 로직 위반 수정")
    print("-" * 40)

    df_fixed = df.copy()
    fixes = {}

    # 1. 음수값을 0으로 수정
    non_negative_cols = ['kills', 'knockdowns', 'assists', 'revives', 'heals', 'boosts',
                        'weaponsAcquired', 'vehicleDestroys', 'roadKills', 'teamKills',
                        'DBNOs', 'walkDistance', 'rideDistance', 'swimDistance']

    for col in non_negative_cols:
        if col in df_fixed.columns:
            negative_count = (df_fixed[col] < 0).sum()
            if negative_count > 0:
                df_fixed.loc[df_fixed[col] < 0, col] = 0
                fixes[f'{col}_negative_fixed'] = negative_count
                print(f"🔧 {col}: {negative_count:,}개의 음수값을 0으로 수정")

    # 2. winPlacePerc 범위 수정 (0-1 범위로 클리핑)
    if 'winPlacePerc' in df_fixed.columns:
        out_of_range = ((df_fixed['winPlacePerc'] < 0) | (df_fixed['winPlacePerc'] > 1)).sum()
        if out_of_range > 0:
            df_fixed['winPlacePerc'] = df_fixed['winPlacePerc'].clip(0, 1)
            fixes['winplaceperc_clipped'] = out_of_range
            print(f"🔧 winPlacePerc: {out_of_range:,}개 값을 0-1 범위로 수정")

    # 3. 극단값 처리 (Winsorization)
    extreme_cols = ['kills', 'damageDealt', 'walkDistance', 'rideDistance']
    for col in extreme_cols:
        if col in df_fixed.columns:
            # 99.5 퍼센타일로 극단값 제한
            upper_limit = df_fixed[col].quantile(0.995)
            extreme_count = (df_fixed[col] > upper_limit).sum()
            if extreme_count > 0:
                df_fixed.loc[df_fixed[col] > upper_limit, col] = upper_limit
                fixes[f'{col}_winsorized'] = extreme_count
                print(f"🔧 {col}: {extreme_count:,}개 극단값을 {upper_limit:.1f}로 제한")

    print(f"✅ 총 {len(fixes)}개 유형의 데이터 품질 이슈 수정")
    return df_fixed, fixes

def optimize_memory_usage(df):
    """메모리 사용량 최적화"""
    print(f"\n💾 메모리 사용량 최적화")
    print("-" * 40)

    df_optimized = df.copy()
    memory_before = df_optimized.memory_usage(deep=True).sum() / 1024**2

    # 수치형 데이터 타입 최적화
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

    # 카테고리형 변수 처리
    categorical_candidates = ['matchType']
    for col in categorical_candidates:
        if col in df_optimized.columns:
            if df_optimized[col].nunique() < len(df_optimized) * 0.5:  # 카디널리티가 낮으면
                df_optimized[col] = df_optimized[col].astype('category')

    memory_after = df_optimized.memory_usage(deep=True).sum() / 1024**2
    memory_saved = memory_before - memory_after

    print(f"📊 메모리 사용량: {memory_before:.1f}MB → {memory_after:.1f}MB")
    print(f"💰 절약된 메모리: {memory_saved:.1f}MB ({memory_saved/memory_before*100:.1f}%)")

    return df_optimized
```
### 3. 종합 데이터 정제 파이프라인
``` python
# 3. 종합 데이터 정제 파이프라인

def comprehensive_data_cleaning_pipeline(df, sample_size=None):
    """종합적인 데이터 정제 파이프라인"""
    print("🚀 종합 데이터 정제 파이프라인 시작")
    print("="*60)

    # 샘플링 (메모리 절약)
    if sample_size and len(df) > sample_size:
        print(f"🎯 메모리 절약을 위해 {sample_size:,}행으로 샘플링")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # 단계별 정제
    steps_log = []

    # 1. 초기 품질 검사
    print(f"\n{'='*20} 1단계: 초기 품질 검사 {'='*20}")
    initial_quality = comprehensive_data_quality_check(df)
    game_violations = check_game_logic_violations(df)
    outliers_info = detect_outliers_statistical(df)
    steps_log.append(f"초기 데이터: {df.shape[0]:,}행")

    # 2. 결측치 처리
    print(f"\n{'='*20} 2단계: 결측치 처리 {'='*20}")
    df_cleaned = clean_missing_values(df, strategy='smart')
    steps_log.append(f"결측치 처리 후: {df_cleaned.shape[0]:,}행")

    # 3. 중복 데이터 처리
    print(f"\n{'='*20} 3단계: 중복 데이터 처리 {'='*20}")
    df_cleaned = handle_duplicates(df_cleaned, strategy='remove')
    steps_log.append(f"중복 제거 후: {df_cleaned.shape[0]:,}행")

    # 4. 게임 로직 위반 수정
    print(f"\n{'='*20} 4단계: 게임 로직 위반 수정 {'='*20}")
    df_cleaned, fixes = fix_game_logic_violations(df_cleaned)
    steps_log.append(f"로직 수정 후: {df_cleaned.shape[0]:,}행")

    # 5. 메모리 최적화
    print(f"\n{'='*20} 5단계: 메모리 최적화 {'='*20}")
    df_cleaned = optimize_memory_usage(df_cleaned)
    steps_log.append(f"최적화 후: {df_cleaned.shape[0]:,}행")

    # 6. 최종 품질 검사
    print(f"\n{'='*20} 6단계: 최종 품질 검사 {'='*20}")
    final_quality = comprehensive_data_quality_check(df_cleaned)

    # 정제 요약
    print(f"\n{'='*20} 데이터 정제 완료 요약 {'='*20}")
    for step in steps_log:
        print(f"📋 {step}")

    print(f"\n✅ 최종 정제 데이터: {df_cleaned.shape[0]:,}행 x {df_cleaned.shape[1]}열")

    # 가비지 컬렉션
    del df
    gc.collect()

    return df_cleaned, {
        'initial_quality': initial_quality,
        'game_violations': game_violations,
        'outliers_info': outliers_info,
        'fixes': fixes,
        'final_quality': final_quality,
        'steps_log': steps_log
    }
```
### 4. 실행 함수
``` python
# 4. 실행 함수

def run_phase2_pipeline(df_raw, sample_size=800000):
    """Phase 2 전체 파이프라인 실행"""
    print("🎮 PUBG Phase 2: 데이터 품질 관리 시작!")
    print("="*60)

    start_time = time.time()

    # 메모리 확인
    check_memory_usage()

    # 종합 데이터 정제 실행
    df_cleaned, cleaning_report = comprehensive_data_cleaning_pipeline(
        df_raw, sample_size=sample_size
    )

    # 실행 시간 계산
    execution_time = time.time() - start_time

    print(f"\n⏰ Phase 2 실행 시간: {execution_time:.1f}초")
    print(f"✅ Phase 2 완료! Phase 3 (EDA)로 진행 가능")

    return df_cleaned, cleaning_report

# 실행 예시
print("\n📋 Phase 2 실행 방법:")
print("="*40)
print("# 1. 데이터 로딩 (Phase 1에서 수행)")
print("df_raw = load_pubg_data(sample_size=None)  # 또는 전체 데이터")
print()
print("# 2. Phase 2 실행")
print("df_cleaned, report = run_phase2_pipeline(df_raw, sample_size=800000)")
print()
print("# 3. 결과 확인")
print("print(f'정제된 데이터: {df_cleaned.shape}')")

print("\n🎯 Phase 2 준비 완료!")
print("📊 다음 단계: Phase 3 - 탐색적 데이터 분석 (EDA)")
```
#### 4. 실행 함수 결과

``` bash
📋 Phase 2 실행 방법:
========================================
# 1. 데이터 로딩 (Phase 1에서 수행)
df_raw = load_pubg_data(sample_size=None)  # 또는 전체 데이터

# 2. Phase 2 실행
df_cleaned, report = run_phase2_pipeline(df_raw, sample_size=800000)

# 3. 결과 확인
print(f'정제된 데이터: {df_cleaned.shape}')

🎯 Phase 2 준비 완료!
📊 다음 단계: Phase 3 - 탐색적 데이터 분석 (EDA)
```

### 실행
``` python
# Phase 1에서 로딩한 데이터 사용
df_raw = load_pubg_data(sample_size=None)  # 전체 또는 샘플

# Phase 2 실행
df_cleaned, report = run_phase2_pipeline(df_raw, sample_size=800000)
```
#### 실행 결과

``` bash
📊 PUBG 데이터 로딩 시작...
📁 파일 크기: 629.0MB
💾 메모리 사용량: 15.5% (1.7GB / 12.7GB)
📈 전체 데이터 로딩...
✅ 데이터 로딩 완료: 4,446,966행 x 29열
```
``` bash
🎮 PUBG Phase 2: 데이터 품질 관리 시작!
============================================================
💾 메모리 사용량: 28.8% (3.3GB / 12.7GB)
🚀 종합 데이터 정제 파이프라인 시작
```
``` bash
============================================================
🎯 메모리 절약을 위해 800,000행으로 샘플링
```
``` bash
==================== 1단계: 초기 품질 검사 ====================
📊 종합 데이터 품질 검사
----------------------------------------
📈 데이터 크기: 800,000행 x 29열

💥 결측치 분석:
Empty DataFrame
Columns: [Column, Missing_Count, Missing_Percentage]
Index: []

🔄 중복 행 수: 0개 (0.00%)

📝 데이터 타입 분포:
int64      19
float64     6
object      4
Name: count, dtype: int64

🎮 게임 로직 기반 데이터 검증
----------------------------------------
❌ 킬 수 > 다운 수: 187,390개
⚠️  최대 킬 수: 55, 30킬 이상: 15개

📊 총 2개의 데이터 품질 이슈 발견

📈 통계적 이상치 탐지 (IQR 방법)
----------------------------------------
📊 rideDistance        : 199,765개 ( 25.0%)
📊 assists             : 139,896개 ( 17.5%)
📊 headshotKills       : 134,943개 ( 16.9%)
📊 maxPlace            : 126,033개 ( 15.8%)
📊 numGroups           : 125,441개 ( 15.7%)
📊 longestKill         : 110,638개 ( 13.8%)
📊 revives             : 105,804개 ( 13.2%)
📊 kills               : 92,899개 ( 11.6%)
📊 heals               : 58,582개 (  7.3%)
📊 DBNOs               : 53,107개 (  6.6%)
```
``` bash
==================== 2단계: 결측치 처리 ====================

🧹 결측치 처리 (전략: smart)
----------------------------------------
📍 winPlacePerc 결측치 제거: 0행
✅ 정제 후 데이터 크기: 800,000행
```
``` bash
==================== 3단계: 중복 데이터 처리 ====================

🔄 중복 데이터 처리 (전략: remove)
----------------------------------------
🗑️ 중복 행 제거: 0행
✅ 처리 후 데이터 크기: 800,000행
```
``` bash
==================== 4단계: 게임 로직 위반 수정 ====================

🎮 게임 로직 위반 수정
----------------------------------------
🔧 kills: 3,653개 극단값을 8.0로 제한
🔧 damageDealt: 4,000개 극단값을 922.2로 제한
🔧 walkDistance: 3,994개 극단값을 4835.0로 제한
🔧 rideDistance: 3,998개 극단값을 8068.0로 제한
✅ 총 4개 유형의 데이터 품질 이슈 수정
```
``` bash
==================== 5단계: 메모리 최적화 ====================

💾 메모리 사용량 최적화
----------------------------------------
📊 메모리 사용량: 364.1MB → 199.9MB
💰 절약된 메모리: 164.2MB (45.1%)
```
``` bash
==================== 6단계: 최종 품질 검사 ====================
📊 종합 데이터 품질 검사
----------------------------------------
📈 데이터 크기: 800,000행 x 29열

💥 결측치 분석:
Empty DataFrame
Columns: [Column, Missing_Count, Missing_Percentage]
Index: []

🔄 중복 행 수: 0개 (0.00%)

📝 데이터 타입 분포:
int8        14
float32      6
int16        5
object       3
category     1
Name: count, dtype: int64
```
``` bash
==================== 데이터 정제 완료 요약 ====================
📋 초기 데이터: 800,000행
📋 결측치 처리 후: 800,000행
📋 중복 제거 후: 800,000행
📋 로직 수정 후: 800,000행
📋 최적화 후: 800,000행

✅ 최종 정제 데이터: 800,000행 x 29열

⏰ Phase 2 실행 시간: 15.7초
✅ Phase 2 완료! Phase 3 (EDA)로 진행 가능
```

## Phase 3: 탐색적 데이터 분석 (EDA)

### 1. 플레이어 행동 특성 분류 및 기본 분석
``` python

print("📊 Phase 3: 탐색적 데이터 분석 시작!")
print("="*60)

# 1. 플레이어 행동 특성 분류 및 기본 분석

def categorize_player_features(df):
    """플레이어 행동 특성을 카테고리별로 분류"""
    print("🎯 플레이어 행동 특성 카테고리 분류")
    print("-" * 40)

    # 특성 카테고리 정의
    feature_categories = {
        'Combat': ['kills', 'knockdowns', 'assists', 'damageDealt', 'longestKill',
                  'headshots', 'weaponsAcquired'],
        'Survival': ['heals', 'boosts', 'revives', 'teamKills', 'DBNOs'],
        'Movement': ['walkDistance', 'rideDistance', 'swimDistance'],
        'Interaction': ['vehicleDestroys', 'roadKills'],
        'Target': ['winPlacePerc'],
        'Meta': ['Id', 'groupId', 'matchId', 'matchType', 'numGroups', 'maxPlace', 'rankPoints']
    }

    # 실제 존재하는 컬럼만 필터링
    available_categories = {}
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            available_categories[category] = available_features
            print(f"📋 {category:<12}: {len(available_features)}개 특성 - {', '.join(available_features[:3])}{'...' if len(available_features) > 3 else ''}")

    return available_categories

def basic_statistics_analysis(df, categories):
    """기본 통계 분석"""
    print(f"\n📈 기본 통계 분석")
    print("-" * 40)

    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId']]

    stats_summary = df[numeric_cols].describe().round(2)
    print("📊 주요 통계 지표:")
    print(stats_summary)

    # 카테고리별 평균값
    print(f"\n🎯 카테고리별 평균값:")
    for category, features in categories.items():
        if category in ['Combat', 'Survival', 'Movement']:
            available_features = [f for f in features if f in numeric_cols]
            if available_features:
                category_mean = df[available_features].mean()
                print(f"\n{category} 평균:")
                for feature in available_features:
                    print(f"  {feature:<15}: {category_mean[feature]:.2f}")

    return stats_summary
```

### 2. 고급 시각화 함수들
``` python
# 2. 고급 시각화 함수들

def create_target_distribution_plots(df):
    """타겟 변수 분포 시각화"""
    print(f"\n🎯 타겟 변수 (winPlacePerc) 분포 분석")
    print("-" * 40)

    if 'winPlacePerc' not in df.columns:
        print("❌ winPlacePerc 컬럼이 없습니다.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Win Place Percentage Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. 히스토그램
    axes[0,0].hist(df['winPlacePerc'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Histogram of Win Place Percentage')
    axes[0,0].set_xlabel('Win Place Percentage')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)

    # 2. 박스 플롯
    axes[0,1].boxplot(df['winPlacePerc'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[0,1].set_title('Box Plot of Win Place Percentage')
    axes[0,1].set_ylabel('Win Place Percentage')
    axes[0,1].grid(True, alpha=0.3)

    # 3. 누적 분포
    sorted_data = np.sort(df['winPlacePerc'])
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1,0].plot(sorted_data, y, linewidth=2, color='green')
    axes[1,0].set_title('Cumulative Distribution Function')
    axes[1,0].set_xlabel('Win Place Percentage')
    axes[1,0].set_ylabel('Cumulative Probability')
    axes[1,0].grid(True, alpha=0.3)

    # 4. 밀도 플롯
    axes[1,1].hist(df['winPlacePerc'], bins=50, density=True, alpha=0.7,
                   color='orange', edgecolor='black', label='Histogram')
    # KDE 추가
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df['winPlacePerc'].dropna())
    x_range = np.linspace(0, 1, 100)
    axes[1,1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    axes[1,1].set_title('Probability Density Function')
    axes[1,1].set_xlabel('Win Place Percentage')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 기본 통계
    print(f"📊 winPlacePerc 기본 통계:")
    print(f"  평균: {df['winPlacePerc'].mean():.3f}")
    print(f"  중앙값: {df['winPlacePerc'].median():.3f}")
    print(f"  표준편차: {df['winPlacePerc'].std():.3f}")
    print(f"  왜도: {df['winPlacePerc'].skew():.3f}")
    print(f"  첨도: {df['winPlacePerc'].kurtosis():.3f}")

def create_combat_analysis_plots(df, categories):
    """전투 관련 특성 분석"""
    print(f"\n⚔️ 전투 행동 특성 분석")
    print("-" * 40)

    combat_features = categories.get('Combat', [])
    available_combat = [f for f in combat_features if f in df.columns][:6]  # 상위 6개만

    if len(available_combat) < 2:
        print("❌ 분석할 전투 특성이 부족합니다.")
        return

    # 2x3 서브플롯
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Combat Behavior Analysis', fontsize=16, fontweight='bold')

    for i, feature in enumerate(available_combat):
        row, col = i // 3, i % 3

        # 히스토그램 (로그 스케일)
        data = df[feature][df[feature] > 0]  # 0보다 큰 값만
        if len(data) > 0:
            axes[row, col].hist(data, bins=50, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
            axes[row, col].set_title(f'{feature} Distribution (>0)')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')

            # 로그 스케일 적용 (극단값이 많은 경우)
            if data.max() / data.mean() > 10:
                axes[row, col].set_yscale('log')
                axes[row, col].set_title(f'{feature} Distribution (Log Scale)')

        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 전투 특성 상관관계
    if len(available_combat) >= 2:
        combat_corr = df[available_combat].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(combat_corr, dtype=bool))
        sns.heatmap(combat_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Combat Features Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

def create_performance_group_analysis(df):
    """성과 그룹별 행동 패턴 분석"""
    print(f"\n🏆 성과 그룹별 행동 패턴 분석")
    print("-" * 40)

    if 'winPlacePerc' not in df.columns:
        print("❌ winPlacePerc 컬럼이 없습니다.")
        return

    # 성과 그룹 생성
    df_analysis = df.copy()
    df_analysis['performance_group'] = pd.cut(df_analysis['winPlacePerc'],
                                            bins=[0, 0.2, 0.5, 0.8, 1.0],
                                            labels=['Low (0-20%)', 'Medium (20-50%)',
                                                   'High (50-80%)', 'Top (80-100%)'])

    # 그룹별 분포
    group_counts = df_analysis['performance_group'].value_counts()
    print("📊 성과 그룹 분포:")
    for group, count in group_counts.items():
        pct = (count / len(df_analysis)) * 100
        print(f"  {group}: {count:,}명 ({pct:.1f}%)")

    # 주요 특성별 그룹 비교
    key_features = ['kills', 'damageDealt', 'walkDistance', 'heals', 'boosts']
    available_features = [f for f in key_features if f in df.columns]

    if available_features:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Group Comparison', fontsize=16, fontweight='bold')

        for i, feature in enumerate(available_features[:6]):
            row, col = i // 3, i % 3

            # 박스 플롯
            df_analysis.boxplot(column=feature, by='performance_group', ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by Performance Group')
            axes[row, col].set_xlabel('Performance Group')
            axes[row, col].set_ylabel(feature)
            axes[row, col].tick_params(axis='x', rotation=45)

        # 빈 서브플롯 숨기기
        for i in range(len(available_features), 6):
            row, col = i // 3, i % 3
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

    # 그룹별 평균 비교 테이블
    if available_features:
        group_means = df_analysis.groupby('performance_group')[available_features].mean()
        print(f"\n📋 성과 그룹별 평균 비교:")
        print(group_means.round(2))

    return df_analysis

def create_correlation_heatmap(df, categories):
    """전체 특성 상관관계 히트맵"""
    print(f"\n🔗 특성 간 상관관계 분석")
    print("-" * 40)

    # 수치형 컬럼 선택 (ID 컬럼 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', 'rankPoints']]

    if len(analysis_cols) < 2:
        print("❌ 분석할 수치형 특성이 부족합니다.")
        return

    # 상관관계 계산
    correlation_matrix = df[analysis_cols].corr()

    # 큰 히트맵 생성
    plt.figure(figsize=(16, 14))

    # 마스크 생성 (상삼각 숨기기)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # 히트맵 생성
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8},
                annot_kws={'size': 8})

    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 강한 상관관계 특성 쌍 찾기
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # 강한 상관관계 기준
                strong_correlations.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })

    if strong_correlations:
        print(f"\n💪 강한 상관관계 (|r| > 0.7) 특성 쌍:")
        for item in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {item['feature1']} ↔ {item['feature2']}: {item['correlation']:.3f}")

    return correlation_matrix
```
### 3. 고급 시각화 - 인터랙티브 플롯
``` python
# 3. 고급 시각화 - 인터랙티브 플롯

def create_interactive_scatter_plots(df):
    """인터랙티브 산점도 생성"""
    print(f"\n🎨 인터랙티브 산점도 생성")
    print("-" * 40)

    if 'winPlacePerc' not in df.columns:
        print("❌ winPlacePerc 컬럼이 없습니다.")
        return

    # 주요 특성 선택
    key_features = ['kills', 'damageDealt', 'walkDistance', 'heals']
    available_features = [f for f in key_features if f in df.columns]

    if len(available_features) < 2:
        print("❌ 분석할 특성이 부족합니다.")
        return

    # 샘플링 (성능을 위해)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # Plotly 산점도
    for i, feature in enumerate(available_features[:3]):  # 상위 3개만
        fig = px.scatter(
            df_sample,
            x=feature,
            y='winPlacePerc',
            color='winPlacePerc',
            title=f'{feature} vs Win Place Percentage',
            labels={feature: feature, 'winPlacePerc': 'Win Place Percentage'},
            color_continuous_scale='viridis',
            opacity=0.6
        )

        fig.update_layout(
            width=800,
            height=600,
            title_font_size=16
        )

        fig.show()

def create_feature_distribution_comparison(df):
    """특성 분포 비교 (상위 vs 하위 성능)"""
    print(f"\n📊 상위/하위 성능자 특성 분포 비교")
    print("-" * 40)

    if 'winPlacePerc' not in df.columns:
        print("❌ winPlacePerc 컬럼이 없습니다.")
        return

    # 상위 10%와 하위 10% 분리
    top_players = df[df['winPlacePerc'] >= df['winPlacePerc'].quantile(0.9)]
    bottom_players = df[df['winPlacePerc'] <= df['winPlacePerc'].quantile(0.1)]

    print(f"🔝 상위 10%: {len(top_players):,}명")
    print(f"🔻 하위 10%: {len(bottom_players):,}명")

    # 주요 특성 비교
    key_features = ['kills', 'damageDealt', 'walkDistance', 'heals', 'boosts', 'weaponsAcquired']
    available_features = [f for f in key_features if f in df.columns]

    if available_features:
        # 비교 테이블 생성
        comparison_stats = pd.DataFrame({
            'Top_10%_Mean': top_players[available_features].mean(),
            'Bottom_10%_Mean': bottom_players[available_features].mean(),
            'Top_10%_Median': top_players[available_features].median(),
            'Bottom_10%_Median': bottom_players[available_features].median()
        })

        comparison_stats['Ratio_Mean'] = comparison_stats['Top_10%_Mean'] / comparison_stats['Bottom_10%_Mean']
        comparison_stats = comparison_stats.round(3)

        print(f"\n📋 상위/하위 10% 성능자 특성 비교:")
        print(comparison_stats)

        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Top 10% vs Bottom 10% Players Feature Comparison', fontsize=16, fontweight='bold')

        for i, feature in enumerate(available_features[:6]):
            row, col = i // 3, i % 3

            # 히스토그램 오버레이
            axes[row, col].hist(bottom_players[feature], bins=30, alpha=0.5,
                               label='Bottom 10%', color='red', density=True)
            axes[row, col].hist(top_players[feature], bins=30, alpha=0.5,
                               label='Top 10%', color='blue', density=True)

            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        # 빈 서브플롯 숨기기
        for i in range(len(available_features), 6):
            row, col = i // 3, i % 3
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

    return top_players, bottom_players, comparison_stats
```
### 4. 종합 EDA 파이프라인
``` python
# 4. 종합 EDA 파이프라인

def comprehensive_eda_pipeline(df):
    """종합적인 탐색적 데이터 분석 실행"""
    print("🚀 종합 탐색적 데이터 분석 시작")
    print("="*60)

    start_time = time.time()
    eda_results = {}

    # 1. 특성 카테고리 분류
    print(f"\n{'='*20} 1단계: 특성 카테고리 분류 {'='*20}")
    categories = categorize_player_features(df)
    eda_results['categories'] = categories

    # 2. 기본 통계 분석
    print(f"\n{'='*20} 2단계: 기본 통계 분석 {'='*20}")
    stats_summary = basic_statistics_analysis(df, categories)
    eda_results['basic_stats'] = stats_summary

    # 3. 타겟 변수 분포 분석
    print(f"\n{'='*20} 3단계: 타겟 변수 분포 분석 {'='*20}")
    create_target_distribution_plots(df)

    # 4. 전투 행동 분석
    print(f"\n{'='*20} 4단계: 전투 행동 분석 {'='*20}")
    create_combat_analysis_plots(df, categories)

    # 5. 성과 그룹별 분석
    print(f"\n{'='*20} 5단계: 성과 그룹별 분석 {'='*20}")
    df_with_groups = create_performance_group_analysis(df)
    eda_results['performance_groups'] = df_with_groups

    # 6. 상관관계 분석
    print(f"\n{'='*20} 6단계: 상관관계 분석 {'='*20}")
    correlation_matrix = create_correlation_heatmap(df, categories)
    eda_results['correlation_matrix'] = correlation_matrix

    # 7. 인터랙티브 시각화
    print(f"\n{'='*20} 7단계: 인터랙티브 시각화 {'='*20}")
    create_interactive_scatter_plots(df)

    # 8. 성능 비교 분석
    print(f"\n{'='*20} 8단계: 성능 비교 분석 {'='*20}")
    top_players, bottom_players, comparison_stats = create_feature_distribution_comparison(df)
    eda_results['performance_comparison'] = {
        'top_players': top_players,
        'bottom_players': bottom_players,
        'comparison_stats': comparison_stats
    }

    # 실행 시간 계산
    execution_time = time.time() - start_time

    print(f"\n{'='*20} EDA 완료 요약 {'='*20}")
    print(f"⏰ 실행 시간: {execution_time:.1f}초")
    print(f"📊 분석된 특성 수: {len([col for col in df.select_dtypes(include=[np.number]).columns if col not in ['Id', 'groupId', 'matchId']])}")
    print(f"🎯 성과 그룹: 4개 (Low, Medium, High, Top)")
    print(f"✅ Phase 3 완료! Phase 4 (특성 엔지니어링)로 진행 가능")

    return eda_results
```
### 5. 실행 함수
``` python
# 5. 실행 함수

def run_phase3_pipeline(df_cleaned):
    """Phase 3 전체 파이프라인 실행"""
    print("🎮 PUBG Phase 3: 탐색적 데이터 분석 시작!")
    print("="*60)

    # 메모리 확인
    check_memory_usage()

    # 종합 EDA 실행
    eda_results = comprehensive_eda_pipeline(df_cleaned)

    return eda_results

# 실행 예시
print("\n📋 Phase 3 실행 방법:")
print("="*40)
print("# Phase 2에서 정제된 데이터 사용")
print("eda_results = run_phase3_pipeline(df_cleaned)")
print()
print("# 결과 확인")
print("print('EDA 결과:', list(eda_results.keys()))")

print("\n🎯 Phase 3 준비 완료!")
print("📊 다음 단계: Phase 4 - 특성 엔지니어링")
```
#### 5. 실행 함수 결과

``` bash
📋 Phase 3 실행 방법:
========================================
# Phase 2에서 정제된 데이터 사용
eda_results = run_phase3_pipeline(df_cleaned)

# 결과 확인
print('EDA 결과:', list(eda_results.keys()))

🎯 Phase 3 준비 완료!
📊 다음 단계: Phase 4 - 특성 엔지니어링
```

### 실행
``` python
# Phase 2에서 정제된 데이터로 EDA 실행
eda_results = run_phase3_pipeline(df_cleaned)

# 결과 확인
print('EDA 결과:', list(eda_results.keys()))
```
#### 실행 결과

``` bash
🎮 PUBG Phase 3: 탐색적 데이터 분석 시작!
============================================================
💾 메모리 사용량: 28.6% (3.3GB / 12.7GB)
🚀 종합 탐색적 데이터 분석 시작
============================================================

==================== 1단계: 특성 카테고리 분류 ====================
🎯 플레이어 행동 특성 카테고리 분류
----------------------------------------
📋 Combat      : 5개 특성 - kills, assists, damageDealt...
📋 Survival    : 5개 특성 - heals, boosts, revives...
📋 Movement    : 3개 특성 - walkDistance, rideDistance, swimDistance
📋 Interaction : 2개 특성 - vehicleDestroys, roadKills
📋 Target      : 1개 특성 - winPlacePerc
📋 Meta        : 7개 특성 - Id, groupId, matchId...
```

``` bash
==================== 2단계: 기본 통계 분석 ====================

📈 기본 통계 분석
----------------------------------------
📊 주요 통계 지표:
         assists     boosts  damageDealt      DBNOs  headshotKills      heals  \
count  800000.00  800000.00    800000.00  800000.00      800000.00  800000.00   
mean        0.23       1.10       129.19       0.66           0.23       1.37   
std         0.59       1.71       161.11       1.15           0.60       2.68   
min         0.00       0.00         0.00       0.00           0.00       0.00   
25%         0.00       0.00         0.00       0.00           0.00       0.00   
50%         0.00       0.00        84.00       0.00           0.00       0.00   
75%         0.00       2.00       185.80       1.00           0.00       2.00   
max        13.00      33.00       922.20      39.00          23.00      63.00   

       killPlace  killPoints      kills  killStreaks  ...    revives  \
count  800000.00   800000.00  800000.00    800000.00  ...  800000.00   
mean       47.64      504.77       0.91         0.54  ...       0.16   
std        27.47      627.37       1.46         0.71  ...       0.47   
min         1.00        0.00       0.00         0.00  ...       0.00   
25%        24.00        0.00       0.00         0.00  ...       0.00   
50%        47.00        0.00       0.00         0.00  ...       0.00   
75%        71.00     1171.00       1.00         1.00  ...       0.00   
max       100.00     2170.00       8.00        18.00  ...      23.00   

       rideDistance  roadKills  swimDistance  teamKills  vehicleDestroys  \
count     800000.00  800000.00     800000.00  800000.00        800000.00   
mean         596.45       0.00          4.50       0.02             0.01   
std         1445.07       0.08         30.65       0.17             0.09   
min            0.00       0.00          0.00       0.00             0.00   
25%            0.00       0.00          0.00       0.00             0.00   
50%            0.00       0.00          0.00       0.00             0.00   
75%            0.14       0.00          0.00       0.00             0.00   
max         8068.00      18.00       3823.00       4.00             4.00   

       walkDistance  weaponsAcquired  winPoints  winPlacePerc  
count     800000.00        800000.00  800000.00     800000.00  
mean        1147.40             3.66     606.26          0.47  
std         1164.22             2.46     739.67          0.31  
min            0.00             0.00       0.00          0.00  
25%          154.80             2.00       0.00          0.20  
50%          682.60             3.00       0.00          0.46  
75%         1971.00             5.00    1495.00          0.74  
max         4835.00           153.00    2013.00          1.00  

[8 rows x 25 columns]

🎯 카테고리별 평균값:

Combat 평균:
  kills          : 0.91
  assists        : 0.23
  damageDealt    : 129.19
  longestKill    : 22.95
  weaponsAcquired: 3.66

Survival 평균:
  heals          : 1.37
  boosts         : 1.10
  revives        : 0.16
  teamKills      : 0.02
  DBNOs          : 0.66

Movement 평균:
  walkDistance   : 1147.40
  rideDistance   : 596.45
  swimDistance   : 4.50
```

``` bash
==================== 3단계: 타겟 변수 분포 분석 ====================

🎯 타겟 변수 (winPlacePerc) 분포 분석
----------------------------------------
```

[!ScreenShot](image/3단계_타겟_변수_분포_분석.png)

``` bash
📊 winPlacePerc 기본 통계:
  평균: 0.473
  중앙값: 0.458
  표준편차: 0.307
  왜도: 0.100
  첨도: -1.245
```

``` bash
==================== 4단계: 전투 행동 분석 ====================

⚔️ 전투 행동 특성 분석
----------------------------------------
```

[!ScreenShot](image/4단계_전투_행동_분석.png)
[!ScreenShot](image/4단계_전투_행동_분석2.png)

``` bash
==================== 5단계: 성과 그룹별 분석 ====================

🏆 성과 그룹별 행동 패턴 분석
----------------------------------------
📊 성과 그룹 분포:
  Medium (20-50%): 236,390명 (29.5%)
  High (50-80%): 203,204명 (25.4%)
  Top (80-100%): 161,073명 (20.1%)
  Low (0-20%): 159,570명 (19.9%)
```

[!ScreenShot](image/5단계_성과_그룹별_분석.png)

``` bash
📋 성과 그룹별 평균 비교:
                   kills  damageDealt  walkDistance  heals  boosts
performance_group                                                 
Low (0-20%)         0.29    56.910000    143.429993   0.13    0.07
Medium (20-50%)     0.59    94.580002    550.049988   0.69    0.40
High (50-80%)       1.00   142.610001   1659.219971   1.96    1.40
Top (80-100%)       2.06   259.019989   2645.280029   3.17    3.06
```

``` bash
==================== 6단계: 상관관계 분석 ====================

🔗 특성 간 상관관계 분석
----------------------------------------
```

[!ScreenShot](image/6단계_상관관계_분석.png)

``` bash
💪 강한 상관관계 (|r| > 0.7) 특성 쌍:
  maxPlace ↔ numGroups: 0.998
  killPoints ↔ winPoints: 0.983
  damageDealt ↔ kills: 0.881
  kills ↔ killStreaks: 0.829
  walkDistance ↔ winPlacePerc: 0.818
  killPlace ↔ killStreaks: -0.810
  killPlace ↔ kills: -0.765
  damageDealt ↔ DBNOs: 0.735
  damageDealt ↔ killStreaks: 0.720
  killPlace ↔ winPlacePerc: -0.719
  DBNOs ↔ kills: 0.712
  damageDealt ↔ killPlace: -0.702
```

``` bash
==================== 7단계: 인터랙티브 시각화 ====================

🎨 인터랙티브 산점도 생성
----------------------------------------
```

``` bash
==================== 8단계: 성능 비교 분석 ====================

📊 상위/하위 성능자 특성 분포 비교
----------------------------------------
🔝 상위 10%: 80,687명
🔻 하위 10%: 80,133명

📋 상위/하위 10% 성능자 특성 비교:
                 Top_10%_Mean  Bottom_10%_Mean  Top_10%_Median  \
kills                   2.550            0.142             2.0   
damageDealt           312.465           35.800           261.3   
walkDistance         2800.155           58.581          2838.0   
heals                   3.377            0.032             2.0   
boosts                  3.574            0.016             3.0   
weaponsAcquired         5.491            1.107             5.0   

                 Bottom_10%_Median  Ratio_Mean  
kills                         0.00      17.944  
damageDealt                   0.00       8.728  
walkDistance                 31.02      47.799  
heals                         0.00     106.946  
boosts                        0.00     220.502  
weaponsAcquired               1.00       4.960  
```

[!ScreenShot](image/8단계_성능_비교_분석.png)

``` bash
==================== EDA 완료 요약 ====================
⏰ 실행 시간: 17.8초
📊 분석된 특성 수: 25
🎯 성과 그룹: 4개 (Low, Medium, High, Top)
✅ Phase 3 완료! Phase 4 (특성 엔지니어링)로 진행 가능
EDA 결과: ['categories', 'basic_stats', 'performance_groups', 'correlation_matrix', 'performance_comparison']
```

## Phase 4: 특성 엔지니어링 (Feature Engineering)

### 0. 진행도 추적 시스템

``` python

print("🔧 Phase 4: 특성 엔지니어링 시작!")
print("="*60)

import time
from datetime import datetime

class ProgressTracker:
    """진행도 추적 클래스"""

    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []

    def update(self, step_name="", details=""):
        """진행도 업데이트"""
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time

        if len(self.step_times) > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time
            eta_str = f"ETA: {eta:.0f}초"
        else:
            eta_str = "ETA: 계산중..."

        progress_pct = (self.current_step / self.total_steps) * 100
        progress_bar = "█" * int(progress_pct // 5) + "░" * (20 - int(progress_pct // 5))

        step_time = elapsed - sum(self.step_times) if self.step_times else elapsed
        self.step_times.append(step_time)

        print(f"\r[{progress_bar}] {progress_pct:.1f}% | {self.current_step}/{self.total_steps} | "
              f"{step_name} | {details} | {eta_str} | 경과: {elapsed:.0f}초", end="")

        if self.current_step == self.total_steps:
            print(f"\n✅ 완료! 총 소요시간: {elapsed:.1f}초")

def show_memory_progress():
    """메모리 사용량 표시"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 메모리: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    except:
        print("💾 메모리 정보 확인 불가")
```
### 1. 파생 특성 생성 함수들
``` python
def create_efficiency_features(df):
    """효율성 지표 특성 생성"""
    print("⚡ 효율성 지표 특성 생성")
    print("-" * 40)

    df_enhanced = df.copy()
    new_features = []

    # 1. 킬 효율성 지표
    if 'kills' in df.columns and 'damageDealt' in df.columns:
        # 데미지당 킬 수 (킬 효율성)
        df_enhanced['kill_efficiency'] = df_enhanced['kills'] / (df_enhanced['damageDealt'] + 1)
        new_features.append('kill_efficiency')

        # 킬당 평균 데미지
        df_enhanced['damage_per_kill'] = df_enhanced['damageDealt'] / (df_enhanced['kills'] + 1)
        new_features.append('damage_per_kill')

    # 2. 이동 효율성 지표
    distance_cols = ['walkDistance', 'rideDistance', 'swimDistance']
    available_distance = [col for col in distance_cols if col in df.columns]

    if len(available_distance) >= 2:
        # 총 이동 거리
        df_enhanced['total_distance'] = df_enhanced[available_distance].sum(axis=1)
        new_features.append('total_distance')

        # 이동 거리당 킬 수
        if 'kills' in df.columns:
            df_enhanced['kills_per_distance'] = df_enhanced['kills'] / (df_enhanced['total_distance'] + 1)
            new_features.append('kills_per_distance')

        # 이동 거리당 데미지
        if 'damageDealt' in df.columns:
            df_enhanced['damage_per_distance'] = df_enhanced['damageDealt'] / (df_enhanced['total_distance'] + 1)
            new_features.append('damage_per_distance')

    # 3. 생존 효율성 지표
    survival_cols = ['heals', 'boosts']
    available_survival = [col for col in survival_cols if col in df.columns]

    if len(available_survival) >= 2:
        # 총 생존 아이템 사용
        df_enhanced['total_heals'] = df_enhanced[available_survival].sum(axis=1)
        new_features.append('total_heals')

        # 생존 아이템당 생존 시간 (대리 지표)
        if 'total_distance' in df_enhanced.columns:
            df_enhanced['heal_efficiency'] = df_enhanced['total_distance'] / (df_enhanced['total_heals'] + 1)
            new_features.append('heal_efficiency')

    # 4. 전투 참여도
    combat_cols = ['kills', 'assists', 'knockdowns']
    available_combat = [col for col in combat_cols if col in df.columns]

    if len(available_combat) >= 2:
        # 총 전투 참여
        df_enhanced['total_combat'] = df_enhanced[available_combat].sum(axis=1)
        new_features.append('total_combat')

        # 전투 참여 비율
        if 'kills' in df.columns:
            df_enhanced['kill_participation'] = df_enhanced['kills'] / (df_enhanced['total_combat'] + 1)
            new_features.append('kill_participation')

    print(f"✅ 생성된 효율성 특성: {len(new_features)}개")
    for feature in new_features:
        print(f"  📊 {feature}")

    return df_enhanced, new_features

def create_ratio_features(df):
    """비율 기반 특성 생성"""
    print("\n📊 비율 기반 특성 생성")
    print("-" * 40)

    df_enhanced = df.copy()
    new_features = []

    # 1. 헤드샷 비율
    if 'headshots' in df.columns and 'kills' in df.columns:
        df_enhanced['headshot_rate'] = df_enhanced['headshots'] / (df_enhanced['kills'] + 1)
        new_features.append('headshot_rate')

    # 2. 어시스트 비율
    if 'assists' in df.columns and 'kills' in df.columns:
        df_enhanced['assist_rate'] = df_enhanced['assists'] / (df_enhanced['kills'] + df_enhanced['assists'] + 1)
        new_features.append('assist_rate')

    # 3. 이동 방식 비율
    distance_cols = ['walkDistance', 'rideDistance', 'swimDistance']
    available_distance = [col for col in distance_cols if col in df.columns]

    if 'total_distance' in df_enhanced.columns:
        for col in available_distance:
            ratio_name = f'{col}_ratio'
            df_enhanced[ratio_name] = df_enhanced[col] / (df_enhanced['total_distance'] + 1)
            new_features.append(ratio_name)

    # 4. 생존 아이템 비율
    if 'heals' in df.columns and 'boosts' in df.columns:
        df_enhanced['heal_boost_ratio'] = df_enhanced['heals'] / (df_enhanced['heals'] + df_enhanced['boosts'] + 1)
        new_features.append('heal_boost_ratio')

    # 5. 무기 습득 효율성
    if 'weaponsAcquired' in df.columns and 'kills' in df.columns:
        df_enhanced['weapons_per_kill'] = df_enhanced['weaponsAcquired'] / (df_enhanced['kills'] + 1)
        new_features.append('weapons_per_kill')

    print(f"✅ 생성된 비율 특성: {len(new_features)}개")
    for feature in new_features:
        print(f"  📊 {feature}")

    return df_enhanced, new_features

def create_game_style_features(df):
    """게임 스타일 지표 생성"""
    print("\n🎮 게임 스타일 지표 생성")
    print("-" * 40)

    df_enhanced = df.copy()
    new_features = []

    # 1. 공격성 지표 (Aggressiveness Score)
    combat_features = ['kills', 'damageDealt', 'longestKill']
    available_combat = [col for col in combat_features if col in df.columns]

    if len(available_combat) >= 2:
        # 표준화된 공격성 점수
        combat_score = 0
        weights = {'kills': 0.4, 'damageDealt': 0.0001, 'longestKill': 0.01}  # 스케일 조정

        for feature in available_combat:
            if feature in weights:
                combat_score += df_enhanced[feature] * weights[feature]

        df_enhanced['aggressiveness_score'] = combat_score
        new_features.append('aggressiveness_score')

    # 2. 생존성 지표 (Survival Score)
    survival_features = ['heals', 'boosts', 'revives']
    available_survival = [col for col in survival_features if col in df.columns]

    if len(available_survival) >= 2:
        survival_score = df_enhanced[available_survival].sum(axis=1)
        df_enhanced['survival_score'] = survival_score
        new_features.append('survival_score')

    # 3. 이동성 지표 (Mobility Score)
    if 'total_distance' in df_enhanced.columns:
        # 이동 거리를 로그 변환하여 스코어화
        df_enhanced['mobility_score'] = np.log1p(df_enhanced['total_distance'])
        new_features.append('mobility_score')

    # 4. 팀플레이 지표 (Teamplay Score)
    teamplay_features = ['assists', 'revives', 'teamKills']
    available_teamplay = [col for col in teamplay_features if col in df.columns]

    if len(available_teamplay) >= 2:
        # 팀킬은 음수로 처리 (페널티)
        teamplay_score = df_enhanced[['assists', 'revives']].sum(axis=1) if 'assists' in df.columns and 'revives' in df.columns else 0
        if 'teamKills' in df.columns:
            teamplay_score -= df_enhanced['teamKills'] * 2  # 팀킬 페널티
        df_enhanced['teamplay_score'] = teamplay_score
        new_features.append('teamplay_score')

    # 5. 종합 플레이 스타일 (Composite Style)
    style_components = ['aggressiveness_score', 'survival_score', 'mobility_score']
    available_styles = [col for col in style_components if col in df_enhanced.columns]

    if len(available_styles) >= 2:
        # 각 스타일의 표준화된 합
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        style_matrix = scaler.fit_transform(df_enhanced[available_styles])
        df_enhanced['composite_style'] = np.mean(style_matrix, axis=1)
        new_features.append('composite_style')

    print(f"✅ 생성된 게임 스타일 특성: {len(new_features)}개")
    for feature in new_features:
        print(f"  🎯 {feature}")

    return df_enhanced, new_features

def create_advanced_features(df):
    """고급 파생 특성 생성"""
    print("\n🚀 고급 파생 특성 생성")
    print("-" * 40)

    df_enhanced = df.copy()
    new_features = []

    # 1. 플레이어 순위 관련 특성
    if 'numGroups' in df.columns and 'maxPlace' in df.columns:
        # 상대적 그룹 크기
        df_enhanced['relative_group_size'] = df_enhanced['maxPlace'] / df_enhanced['numGroups']
        new_features.append('relative_group_size')

        # 경쟁 강도 (그룹 수 대비 최대 순위)
        df_enhanced['competition_intensity'] = df_enhanced['numGroups'] / df_enhanced['maxPlace']
        new_features.append('competition_intensity')

    # 2. 상호작용 특성 (Feature Interactions)
    if 'kills' in df.columns and 'damageDealt' in df.columns:
        # 킬과 데미지의 상호작용
        df_enhanced['kill_damage_interaction'] = df_enhanced['kills'] * df_enhanced['damageDealt']
        new_features.append('kill_damage_interaction')

    # 3. 로그 변환 특성 (중요한 특성들의 로그 변환)
    log_candidates = ['damageDealt', 'walkDistance', 'rideDistance']
    for feature in log_candidates:
        if feature in df.columns:
            log_feature_name = f'{feature}_log'
            df_enhanced[log_feature_name] = np.log1p(df_enhanced[feature])
            new_features.append(log_feature_name)

    # 4. 이진 특성 (Binary Features)
    # 특정 행동을 했는지 여부
    binary_candidates = ['kills', 'vehicleDestroys', 'roadKills', 'swimDistance']
    for feature in binary_candidates:
        if feature in df.columns:
            binary_feature_name = f'has_{feature}'
            df_enhanced[binary_feature_name] = (df_enhanced[feature] > 0).astype(int)
            new_features.append(binary_feature_name)

    # 5. 구간화 특성 (Binning)
    if 'kills' in df.columns:
        # 킬 수를 구간별로 분류
        df_enhanced['kill_category'] = pd.cut(df_enhanced['kills'],
                                            bins=[-1, 0, 2, 5, 10, float('inf')],
                                            labels=['No_Kill', 'Low_Kill', 'Medium_Kill', 'High_Kill', 'Very_High_Kill'])
        # 원-핫 인코딩
        kill_dummies = pd.get_dummies(df_enhanced['kill_category'], prefix='kill_cat')
        df_enhanced = pd.concat([df_enhanced, kill_dummies], axis=1)
        new_features.extend(kill_dummies.columns.tolist())

    print(f"✅ 생성된 고급 특성: {len(new_features)}개")
    for feature in new_features[:10]:  # 처음 10개만 표시
        print(f"  🔬 {feature}")
    if len(new_features) > 10:
        print(f"  ... 및 {len(new_features)-10}개 추가 특성")

    return df_enhanced, new_features
```
#### 1-1. 빠른 버전 특성 생성 함수들 (진행도 표시용)
``` python
print("\n🚀 빠른 처리를 위한 최적화 함수들")
print("-" * 50)

def create_efficiency_features_fast(df):
    """효율성 지표 특성 생성 (빠른 버전)"""
    df_enhanced = df.copy()
    new_features = []

    # 핵심 특성만 생성
    if 'kills' in df.columns and 'damageDealt' in df.columns:
        df_enhanced['kill_efficiency'] = df_enhanced['kills'] / (df_enhanced['damageDealt'] + 1)
        df_enhanced['damage_per_kill'] = df_enhanced['damageDealt'] / (df_enhanced['kills'] + 1)
        new_features.extend(['kill_efficiency', 'damage_per_kill'])

    distance_cols = ['walkDistance', 'rideDistance', 'swimDistance']
    available_distance = [col for col in distance_cols if col in df.columns]
    if len(available_distance) >= 2:
        df_enhanced['total_distance'] = df_enhanced[available_distance].sum(axis=1)
        new_features.append('total_distance')

    return df_enhanced, new_features

def create_ratio_features_fast(df):
    """비율 기반 특성 생성 (빠른 버전)"""
    df_enhanced = df.copy()
    new_features = []

    if 'heals' in df.columns and 'boosts' in df.columns:
        df_enhanced['total_heals'] = df_enhanced['heals'] + df_enhanced['boosts']
        df_enhanced['heal_boost_ratio'] = df_enhanced['heals'] / (df_enhanced['total_heals'] + 1)
        new_features.extend(['total_heals', 'heal_boost_ratio'])

    return df_enhanced, new_features

def create_game_style_features_fast(df):
    """게임 스타일 지표 생성 (빠른 버전)"""
    df_enhanced = df.copy()
    new_features = []

    if 'kills' in df.columns and 'damageDealt' in df.columns:
        df_enhanced['aggressiveness_score'] = (
            df_enhanced['kills'] * 0.4 +
            df_enhanced['damageDealt'] * 0.001
        )
        new_features.append('aggressiveness_score')

    return df_enhanced, new_features

def create_advanced_features_fast(df):
    """고급 파생 특성 생성 (빠른 버전)"""
    df_enhanced = df.copy()
    new_features = []

    # 로그 변환
    log_candidates = ['damageDealt', 'walkDistance']
    for feature in log_candidates:
        if feature in df.columns:
            df_enhanced[f'{feature}_log'] = np.log1p(df_enhanced[feature])
            new_features.append(f'{feature}_log')

    # 이진 특성
    binary_candidates = ['kills', 'swimDistance']
    for feature in binary_candidates:
        if feature in df.columns:
            df_enhanced[f'has_{feature}'] = (df_enhanced[feature] > 0).astype(int)
            new_features.append(f'has_{feature}')

    return df_enhanced, new_features

print("✅ 빠른 버전 특성 생성 함수들 준비 완료!")
```
### 2. 특성 선택 및 필터링 함수들
``` python
def detect_multicollinearity(df, threshold=0.95):
    """다중공선성 탐지 및 제거"""
    print(f"\n🔗 다중공선성 탐지 (임계값: {threshold})")
    print("-" * 40)

    # 수치형 특성만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', 'winPlacePerc']]

    if len(analysis_cols) < 2:
        print("❌ 분석할 특성이 부족합니다.")
        return df, []

    # 상관관계 매트릭스 계산
    corr_matrix = df[analysis_cols].corr().abs()

    # 상삼각 매트릭스에서 높은 상관관계 찾기
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 제거할 특성 찾기
    high_corr_features = []
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > threshold):
            high_corr_features.append(column)

    print(f"🔍 높은 상관관계 ({threshold} 이상) 특성 쌍:")
    removed_features = []

    for i, column in enumerate(upper_triangle.columns):
        correlated_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
        if correlated_features:
            print(f"  📊 {column} ↔ {correlated_features}")
            # 더 많은 상관관계를 가진 특성 제거
            if column not in removed_features:
                removed_features.append(column)

    # 중복 제거된 특성들 제거
    df_filtered = df.drop(columns=removed_features)

    print(f"🗑️ 제거된 특성: {len(removed_features)}개")
    print(f"✅ 필터링 후 특성 수: {df_filtered.shape[1]}개")

    return df_filtered, removed_features

def select_features_by_importance(df, target_col='winPlacePerc', method='random_forest', top_k=50):
    """특성 중요도 기반 특성 선택"""
    print(f"\n⭐ 특성 중요도 기반 선택 ({method}, Top {top_k})")
    print("-" * 40)

    if target_col not in df.columns:
        print(f"❌ 타겟 컬럼 '{target_col}'이 없습니다.")
        return df, []

    # 특성과 타겟 분리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', target_col]]

    if len(feature_cols) < top_k:
        print(f"⚠️ 사용 가능한 특성({len(feature_cols)})이 요청된 수({top_k})보다 적습니다.")
        top_k = len(feature_cols)

    X = df[feature_cols]
    y = df[target_col]

    # 결측치 처리
    X = X.fillna(0)
    y = y.fillna(y.median())

    if method == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        importances = model.feature_importances_

    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        importances = mutual_info_regression(X, y, random_state=42)

    # 중요도 순으로 정렬
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 상위 특성 선택
    top_features = feature_importance.head(top_k)['feature'].tolist()

    print(f"📊 상위 {min(10, len(top_features))}개 중요 특성:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

    # 선택된 특성과 메타 컬럼으로 데이터 필터링
    meta_cols = ['Id', 'groupId', 'matchId', target_col]
    keep_cols = [col for col in meta_cols if col in df.columns] + top_features
    df_selected = df[keep_cols]

    print(f"✅ 선택된 특성: {len(top_features)}개")

    return df_selected, top_features

def handle_infinite_values(df):
    """무한값 및 극단값 처리"""
    print(f"\n♾️ 무한값 및 극단값 처리")
    print("-" * 40)

    df_cleaned = df.copy()

    # 무한값 탐지
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    infinite_issues = {}

    for col in numeric_cols:
        inf_count = np.isinf(df_cleaned[col]).sum()
        if inf_count > 0:
            infinite_issues[col] = inf_count
            # 무한값을 해당 컬럼의 최대값으로 대체
            finite_values = df_cleaned[col][np.isfinite(df_cleaned[col])]
            if len(finite_values) > 0:
                max_finite = finite_values.max()
                df_cleaned.loc[np.isinf(df_cleaned[col]), col] = max_finite
            print(f"🔧 {col}: {inf_count}개 무한값을 {max_finite:.2f}로 대체")

    # NaN 값 탐지 및 처리
    nan_issues = df_cleaned.isnull().sum()
    nan_issues = nan_issues[nan_issues > 0]

    for col, nan_count in nan_issues.items():
        if col in numeric_cols:
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            print(f"🔧 {col}: {nan_count}개 NaN값을 {median_val:.2f}로 대체")

    if not infinite_issues and len(nan_issues) == 0:
        print("✅ 무한값 및 NaN값이 발견되지 않았습니다.")

    return df_cleaned
```
#### 2-1. 빠른 버전 특성 선택 함수들 (진행도 표시용)
``` python
def detect_multicollinearity_fast(df, threshold=0.95):
    """다중공선성 탐지 (빠른 버전)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', 'winPlacePerc']]

    if len(analysis_cols) < 2:
        return df, []

    # 간단한 상관관계 체크
    corr_matrix = df[analysis_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    df_filtered = df.drop(columns=high_corr_features)
    print(f"🗑️ 제거된 특성: {len(high_corr_features)}개")

    return df_filtered, high_corr_features

def select_features_by_correlation(df, target_col='winPlacePerc', top_k=30):
    """상관관계 기반 특성 선택 (RandomForest 대신 빠른 방법)"""
    if target_col not in df.columns:
        return df, []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', target_col]]

    # 상관관계 계산 (RandomForest 대신)
    target_corr = df[feature_cols + [target_col]].corr()[target_col].abs()
    target_corr = target_corr.drop(target_col).sort_values(ascending=False)

    top_k = min(top_k, len(target_corr))
    top_features = target_corr.head(top_k).index.tolist()

    print(f"⭐ 상위 5개 중요 특성:")
    for i, feature in enumerate(top_features[:5], 1):
        print(f"  {i}. {feature}: {target_corr[feature]:.3f}")

    # 선택된 특성으로 데이터 필터링
    meta_cols = ['Id', 'groupId', 'matchId', target_col]
    keep_cols = [col for col in meta_cols if col in df.columns] + top_features
    df_selected = df[keep_cols]

    return df_selected, top_features

print("✅ 빠른 버전 특성 선택 함수들 준비 완료!")
```
### 3. 특성 스케일링 및 전처리
``` python
def apply_feature_scaling(df, target_col='winPlacePerc', method='standard'):
    """특성 스케일링 적용"""
    print(f"\n⚖️ 특성 스케일링 적용 ({method})")
    print("-" * 40)

    df_scaled = df.copy()

    # 스케일링할 특성 선택 (수치형, 메타 컬럼 제외)
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
    scale_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', target_col]]

    if len(scale_cols) == 0:
        print("❌ 스케일링할 특성이 없습니다.")
        return df_scaled, None

    # 스케일러 선택
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        print(f"❌ 지원하지 않는 스케일링 방법: {method}")
        return df_scaled, None

    # 스케일링 적용
    scaled_data = scaler.fit_transform(df_scaled[scale_cols])
    df_scaled[scale_cols] = scaled_data

    print(f"✅ {len(scale_cols)}개 특성에 {method} 스케일링 적용")
    print(f"📊 스케일링 후 특성 범위 예시:")
    for col in scale_cols[:5]:  # 처음 5개만 표시
        min_val, max_val = df_scaled[col].min(), df_scaled[col].max()
        print(f"  {col:<20}: [{min_val:.3f}, {max_val:.3f}]")

    return df_scaled, scaler

def prepare_modeling_data(df, target_col='winPlacePerc', test_size=0.2, random_state=42):
    """모델링을 위한 데이터 준비"""
    print(f"\n🎯 모델링 데이터 준비")
    print("-" * 40)

    if target_col not in df.columns:
        print(f"❌ 타겟 컬럼 '{target_col}'이 없습니다.")
        return None

    # 특성과 타겟 분리
    feature_cols = [col for col in df.columns if col not in ['Id', 'groupId', 'matchId', target_col]]

    X = df[feature_cols]
    y = df[target_col]

    print(f"📊 특성 수: {X.shape[1]}개")
    print(f"📊 샘플 수: {X.shape[0]:,}개")
    print(f"🎯 타겟 분포:")
    print(f"  평균: {y.mean():.3f}")
    print(f"  표준편차: {y.std():.3f}")
    print(f"  범위: [{y.min():.3f}, {y.max():.3f}]")

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    print(f"✅ 데이터 분할 완료:")
    print(f"  훈련 세트: {X_train.shape[0]:,}개 ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"  테스트 세트: {X_test.shape[0]:,}개 ({X_test.shape[0]/len(df)*100:.1f}%)")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols
    }
```
### 4. 종합 특성 엔지니어링 파이프라인 (진행도 표시 버전)
``` python
def comprehensive_feature_engineering_pipeline(df, target_col='winPlacePerc', sample_size=100000):
    """진행도 표시가 포함된 종합적인 특성 엔지니어링 파이프라인"""
    print("🚀 종합 특성 엔지니어링 파이프라인 시작 (진행도 표시)")
    print("="*60)

    # 진행도 추적기 초기화 (총 9단계)
    progress = ProgressTracker(9, "Phase 4")
    start_time = time.time()
    pipeline_results = {'original_shape': df.shape}

    # 0. 데이터 샘플링 (메모리 절약)
    print("\n🎯 데이터 준비 중...")
    if len(df) > sample_size:
        df_work = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"📊 샘플링: {len(df_work):,}행 (원본: {len(df):,}행)")
    else:
        df_work = df.copy()

    show_memory_progress()

    # 1. 파생 특성 생성
    progress.update("효율성 특성", "킬 효율성, 이동 효율성 계산")
    df_enhanced, efficiency_features = create_efficiency_features_fast(df_work)
    pipeline_results['efficiency_features'] = efficiency_features

    progress.update("비율 특성", f"{len(efficiency_features)}개 효율성 특성 생성됨")
    df_enhanced, ratio_features = create_ratio_features_fast(df_enhanced)
    pipeline_results['ratio_features'] = ratio_features

    progress.update("게임 스타일", f"총 {len(efficiency_features + ratio_features)}개 파생 특성")
    df_enhanced, style_features = create_game_style_features_fast(df_enhanced)
    pipeline_results['style_features'] = style_features

    progress.update("고급 특성", "로그 변환 및 이진 특성 생성")
    df_enhanced, advanced_features = create_advanced_features_fast(df_enhanced)
    pipeline_results['advanced_features'] = advanced_features

    show_memory_progress()

    # 2. 데이터 품질 관리
    progress.update("데이터 정제", f"{len(advanced_features)}개 새 특성 생성 완료")
    df_enhanced = handle_infinite_values(df_enhanced)

    progress.update("특성 선택", "상관관계 기반 특성 필터링")
    df_enhanced, removed_features = detect_multicollinearity_fast(df_enhanced, threshold=0.95)
    pipeline_results['removed_multicollinear'] = removed_features

    # 3. 특성 선택 (빠른 방법)
    progress.update("특성 중요도", "상관관계 기반 중요도 계산")
    df_selected, important_features = select_features_by_correlation(
        df_enhanced, target_col=target_col, top_k=30  # 30개로 제한
    )
    pipeline_results['important_features'] = important_features

    # 4. 특성 스케일링
    progress.update("스케일링", f"{len(important_features)}개 특성 표준화")
    df_scaled, scaler = apply_feature_scaling(df_selected, target_col=target_col, method='standard')
    pipeline_results['scaler'] = scaler

    # 5. 모델링 데이터 준비
    progress.update("데이터 분할", "훈련/테스트 세트 생성")
    modeling_data = prepare_modeling_data(df_scaled, target_col=target_col)
    pipeline_results['modeling_data'] = modeling_data

    total_time = time.time() - start_time

    print(f"\n🎉 Phase 4 완료!")
    print(f"⏰ 총 소요시간: {total_time:.1f}초")
    print(f"📊 원본 특성: {df.shape[1]}개")
    print(f"📈 생성된 특성: {len(efficiency_features + ratio_features + style_features + advanced_features)}개")
    print(f"🗑️ 제거된 특성: {len(removed_features)}개")
    print(f"⭐ 최종 선택된 특성: {len(important_features)}개")
    print(f"🎯 최종 데이터 크기: {df_scaled.shape[0]:,}행 x {df_scaled.shape[1]}열")
    print(f"✅ Phase 4 완료! Phase 5 (클러스터링)로 진행 가능")

    return df_scaled, pipeline_results

# 기존 버전도 유지 (호환성을 위해)
def comprehensive_feature_engineering_pipeline_original(df, target_col='winPlacePerc'):
    """종합적인 특성 엔지니어링 파이프라인 (원본 버전)"""
    print("🚀 종합 특성 엔지니어링 파이프라인 시작")
    print("="*60)

    start_time = time.time()
    pipeline_results = {'original_shape': df.shape}

    # 1. 파생 특성 생성
    print(f"\n{'='*20} 1단계: 효율성 특성 생성 {'='*20}")
    df_enhanced, efficiency_features = create_efficiency_features(df)
    pipeline_results['efficiency_features'] = efficiency_features

    print(f"\n{'='*20} 2단계: 비율 특성 생성 {'='*20}")
    df_enhanced, ratio_features = create_ratio_features(df_enhanced)
    pipeline_results['ratio_features'] = ratio_features

    print(f"\n{'='*20} 3단계: 게임 스타일 특성 생성 {'='*20}")
    df_enhanced, style_features = create_game_style_features(df_enhanced)
    pipeline_results['style_features'] = style_features

    print(f"\n{'='*20} 4단계: 고급 특성 생성 {'='*20}")
    df_enhanced, advanced_features = create_advanced_features(df_enhanced)
    pipeline_results['advanced_features'] = advanced_features

    # 2. 데이터 품질 관리
    print(f"\n{'='*20} 5단계: 무한값 처리 {'='*20}")
    df_enhanced = handle_infinite_values(df_enhanced)

    print(f"\n{'='*20} 6단계: 다중공선성 제거 {'='*20}")
    df_enhanced, removed_features = detect_multicollinearity(df_enhanced, threshold=0.95)
    pipeline_results['removed_multicollinear'] = removed_features

    # 3. 특성 선택
    print(f"\n{'='*20} 7단계: 특성 중요도 선택 {'='*20}")
    df_selected, important_features = select_features_by_importance(
        df_enhanced, target_col=target_col, method='random_forest', top_k=50
    )
    pipeline_results['important_features'] = important_features

    # 4. 특성 스케일링
    print(f"\n{'='*20} 8단계: 특성 스케일링 {'='*20}")
    df_scaled, scaler = apply_feature_scaling(df_selected, target_col=target_col, method='standard')
    pipeline_results['scaler'] = scaler

    # 5. 모델링 데이터 준비
    print(f"\n{'='*20} 9단계: 모델링 데이터 준비 {'='*20}")
    modeling_data = prepare_modeling_data(df_scaled, target_col=target_col)
    pipeline_results['modeling_data'] = modeling_data

    # 실행 시간 계산
    execution_time = time.time() - start_time

    # 최종 요약
    print(f"\n{'='*20} 특성 엔지니어링 완료 요약 {'='*20}")
    print(f"⏰ 실행 시간: {execution_time:.1f}초")
    print(f"📊 원본 특성: {df.shape[1]}개")
    print(f"📈 생성된 특성: {len(efficiency_features + ratio_features + style_features + advanced_features)}개")
    print(f"🗑️ 제거된 특성: {len(removed_features)}개")
    print(f"⭐ 최종 선택된 특성: {len(important_features)}개")
    print(f"🎯 최종 데이터 크기: {df_scaled.shape[0]:,}행 x {df_scaled.shape[1]}열")
    print(f"✅ Phase 4 완료! Phase 5 (클러스터링)로 진행 가능")

    return df_scaled, pipeline_results
```
### 5. 실행 함수
``` python
# 5. 실행 함수

def run_phase4_pipeline(df_cleaned, target_col='winPlacePerc', use_progress=True, sample_size=100000):
    """Phase 4 전체 파이프라인 실행"""
    print("🎮 PUBG Phase 4: 특성 엔지니어링 시작!")
    print("="*60)

    # 메모리 확인
    check_memory_usage()

    # 진행도 표시 버전 또는 원본 버전 선택
    if use_progress:
        print("🚀 진행도 표시 버전으로 실행 (빠른 처리)")
        df_engineered, results = comprehensive_feature_engineering_pipeline(
            df_cleaned, target_col, sample_size=sample_size
        )
    else:
        print("📊 원본 버전으로 실행 (전체 특성)")
        df_engineered, results = comprehensive_feature_engineering_pipeline_original(
            df_cleaned, target_col
        )

    return df_engineered, results
```
### 6. 특성 품질 검증 함수
``` python
# 6. 특성 품질 검증 함수

def validate_engineered_features(df, pipeline_results):
    """생성된 특성의 품질 검증"""
    print("\n🔍 특성 엔지니어링 품질 검증")
    print("="*50)

    # 1. 생성된 특성 통계 요약
    print("📊 생성된 특성 카테고리별 요약:")
    categories = {
        'Efficiency': pipeline_results.get('efficiency_features', []),
        'Ratio': pipeline_results.get('ratio_features', []),
        'Style': pipeline_results.get('style_features', []),
        'Advanced': pipeline_results.get('advanced_features', [])
    }

    for category, features in categories.items():
        if features:
            print(f"  {category:<12}: {len(features):2d}개 특성")

    # 2. 최종 선택된 특성 분석
    important_features = pipeline_results.get('important_features', [])
    if important_features:
        print(f"\n⭐ 최종 선택된 상위 특성 ({len(important_features)}개):")

        # 카테고리별 분류
        for category, features in categories.items():
            selected_in_category = [f for f in features if f in important_features]
            if selected_in_category:
                print(f"  {category} ({len(selected_in_category)}개): {', '.join(selected_in_category[:3])}{'...' if len(selected_in_category) > 3 else ''}")

    # 3. 데이터 품질 지표
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['Id', 'groupId', 'matchId', 'winPlacePerc']]

    if feature_cols:
        print(f"\n📈 최종 특성 품질 지표:")
        quality_stats = df[feature_cols].describe()

        # 결측치 확인
        missing_count = df[feature_cols].isnull().sum().sum()
        print(f"  결측치: {missing_count}개")

        # 무한값 확인
        inf_count = np.isinf(df[feature_cols]).sum().sum()
        print(f"  무한값: {inf_count}개")

        # 변동성 확인 (표준편차가 0인 특성)
        zero_var_features = quality_stats.loc['std'][quality_stats.loc['std'] == 0]
        print(f"  변동성 없는 특성: {len(zero_var_features)}개")

        if len(zero_var_features) > 0:
            print(f"    - {', '.join(zero_var_features.index.tolist())}")

    # 4. 모델링 준비 상태 확인
    modeling_data = pipeline_results.get('modeling_data')
    if modeling_data:
        print(f"\n🎯 모델링 준비 상태:")
        print(f"  훈련 세트: {modeling_data['X_train'].shape}")
        print(f"  테스트 세트: {modeling_data['X_test'].shape}")
        print(f"  특성 수: {len(modeling_data['feature_names'])}개")
        print(f"  타겟 범위: [{modeling_data['y_train'].min():.3f}, {modeling_data['y_train'].max():.3f}]")

    print(f"\n✅ 특성 엔지니어링 품질 검증 완료")
```
### 7. 실행 예시 및 가이드
``` python
def display_feature_engineering_guide():
    """특성 엔지니어링 실행 가이드"""
    print("\n📋 Phase 4 실행 방법:")
    print("="*50)
    print("# 1. 빠른 버전 (진행도 표시, 권장)")
    print("df_engineered, results = run_phase4_pipeline(df_cleaned, use_progress=True, sample_size=100000)")
    print()
    print("# 2. 원본 버전 (전체 특성, 시간 오래 걸림)")
    print("df_engineered, results = run_phase4_pipeline(df_cleaned, use_progress=False)")
    print()
    print("# 3. 특성 품질 검증")
    print("validate_engineered_features(df_engineered, results)")
    print()
    print("# 4. 모델링 데이터 추출")
    print("modeling_data = results['modeling_data']")
    print("X_train = modeling_data['X_train']")
    print("y_train = modeling_data['y_train']")
    print()
    print("# 5. 다음 단계로 진행")
    print("# Phase 5: 클러스터링 (비지도 학습)")
    print("# Phase 6: 딥러닝 분류 모델링")

# 가이드 표시
display_feature_engineering_guide()

print("\n🎯 Phase 4 진행도 표시 완전판 준비 완료!")
print("🔧 다음 단계: Phase 5 - 비지도 학습 (클러스터링)")
```
#### 7. 실행 예시 및 가이드 결과

``` bash
📋 Phase 4 실행 방법:
==================================================
# 1. 빠른 버전 (진행도 표시, 권장)
df_engineered, results = run_phase4_pipeline(df_cleaned, use_progress=True, sample_size=100000)

# 2. 원본 버전 (전체 특성, 시간 오래 걸림)
df_engineered, results = run_phase4_pipeline(df_cleaned, use_progress=False)

# 3. 특성 품질 검증
validate_engineered_features(df_engineered, results)

# 4. 모델링 데이터 추출
modeling_data = results['modeling_data']
X_train = modeling_data['X_train']
y_train = modeling_data['y_train']

# 5. 다음 단계로 진행
# Phase 5: 클러스터링 (비지도 학습)
# Phase 6: 딥러닝 분류 모델링

🎯 Phase 4 진행도 표시 완전판 준비 완료!
🔧 다음 단계: Phase 5 - 비지도 학습 (클러스터링)
```

### 8. 사용 예시
``` python
print("\n📝 사용 예시:")
print("="*30)
print("# 빠른 실행 (권장)")
print("df_engineered, results = run_phase4_pipeline(df_cleaned)")
print()
print("# 결과 확인")
print("print('최종 특성 수:', len(results['important_features']))")
print("print('모델링 데이터:', results['modeling_data']['X_train'].shape)")
```
#### 8. 사용 예시 결과

``` bash
📝 사용 예시:
==============================
# 빠른 실행 (권장)
df_engineered, results = run_phase4_pipeline(df_cleaned)

# 결과 확인
print('최종 특성 수:', len(results['important_features']))
print('모델링 데이터:', results['modeling_data']['X_train'].shape)
```

### 실행
``` python
# Phase 4 전체 파이프라인 실행
df_engineered, results = run_phase4_pipeline(df_cleaned)

# 특성 품질 검증
validate_engineered_features(df_engineered, results)

# 모델링 데이터 추출
modeling_data = results['modeling_data']
X_train, y_train = modeling_data['X_train'], modeling_data['y_train']
```

#### 실행 결과

``` bash
🎮 PUBG Phase 4: 특성 엔지니어링 시작!
============================================================
💾 메모리 사용량: 30.2% (3.5GB / 12.7GB)
🚀 진행도 표시 버전으로 실행 (빠른 처리)
🚀 종합 특성 엔지니어링 파이프라인 시작 (진행도 표시)
============================================================

🎯 데이터 준비 중...
📊 샘플링: 100,000행 (원본: 800,000행)
💾 메모리: 30.0% (3.5GB/12.7GB)
[████████░░░░░░░░░░░░] 44.4% | 4/9 | 고급 특성 | 로그 변환 및 이진 특성 생성 | ETA: 1초 | 경과: 0초💾 메모리: 30.0% (3.5GB/12.7GB)
[███████████░░░░░░░░░] 55.6% | 5/9 | 데이터 정제 | 4개 새 특성 생성 완료 | ETA: 0초 | 경과: 0초
♾️ 무한값 및 극단값 처리
----------------------------------------
✅ 무한값 및 NaN값이 발견되지 않았습니다.
[█████████████░░░░░░░] 66.7% | 6/9 | 특성 선택 | 상관관계 기반 특성 필터링 | ETA: 0초 | 경과: 1초🗑️ 제거된 특성: 4개
[███████████████░░░░░] 77.8% | 7/9 | 특성 중요도 | 상관관계 기반 중요도 계산 | ETA: 0초 | 경과: 1초⭐ 상위 5개 중요 특성:
  1. walkDistance: 0.817
  2. walkDistance_log: 0.794
  3. killPlace: 0.721
  4. total_distance: 0.685
  5. boosts: 0.633
[█████████████████░░░] 88.9% | 8/9 | 스케일링 | 30개 특성 표준화 | ETA: 0초 | 경과: 1초
⚖️ 특성 스케일링 적용 (standard)
----------------------------------------
✅ 30개 특성에 standard 스케일링 적용
📊 스케일링 후 특성 범위 예시:
  walkDistance        : [-0.985, 3.154]
  walkDistance_log    : [-3.360, 1.306]
  killPlace           : [-1.700, 1.904]
  total_distance      : [-0.820, 5.237]
  boosts              : [-0.643, 8.626]
[████████████████████] 100.0% | 9/9 | 데이터 분할 | 훈련/테스트 세트 생성 | ETA: 0초 | 경과: 2초
✅ 완료! 총 소요시간: 1.5초

🎯 모델링 데이터 준비
----------------------------------------
📊 특성 수: 30개
📊 샘플 수: 100,000개
🎯 타겟 분포:
  평균: 0.473
  표준편차: 0.308
  범위: [0.000, 1.000]
✅ 데이터 분할 완료:
  훈련 세트: 80,000개 (80.0%)
  테스트 세트: 20,000개 (20.0%)

🎉 Phase 4 완료!
⏰ 총 소요시간: 1.6초
📊 원본 특성: 29개
📈 생성된 특성: 10개
🗑️ 제거된 특성: 4개
⭐ 최종 선택된 특성: 30개
🎯 최종 데이터 크기: 100,000행 x 34열
✅ Phase 4 완료! Phase 5 (클러스터링)로 진행 가능

🔍 특성 엔지니어링 품질 검증
==================================================
📊 생성된 특성 카테고리별 요약:
  Efficiency  :  3개 특성
  Ratio       :  2개 특성
  Style       :  1개 특성
  Advanced    :  4개 특성

⭐ 최종 선택된 상위 특성 (30개):
  Efficiency (3개): kill_efficiency, damage_per_kill, total_distance
  Ratio (2개): total_heals, heal_boost_ratio
  Advanced (4개): damageDealt_log, walkDistance_log, has_kills...

📈 최종 특성 품질 지표:
  결측치: 0개
  무한값: 0개
  변동성 없는 특성: 0개

🎯 모델링 준비 상태:
  훈련 세트: (80000, 30)
  테스트 세트: (20000, 30)
  특성 수: 30개
  타겟 범위: [0.000, 1.000]

✅ 특성 엔지니어링 품질 검증 완료
```

## Phase 5: 비지도 학습 (클러스터링)

### 0. 진행도 표시 유틸리티 함수들
``` python

import time
from tqdm.auto import tqdm
import sys
from IPython.display import display, HTML, clear_output

# 진행도 표시 관련 함수들
def show_progress_bar(current, total, description="Processing", bar_length=50):
    """진행도 바 표시"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    percentage = progress * 100
    print(f'\r{description}: |{bar}| {percentage:.1f}% ({current}/{total})', end='', flush=True)

    if current == total:
        print()  # 완료 시 줄바꿈

def create_phase_tracker():
    """전체 Phase 5 진행 상황 추적기"""
    phases = {
        1: {"name": "최적 클러스터 수 탐색", "status": "대기중", "progress": 0},
        2: {"name": "K-Means 클러스터링 실행", "status": "대기중", "progress": 0},
        3: {"name": "클러스터 특성 분석", "status": "대기중", "progress": 0},
        4: {"name": "클러스터 시각화", "status": "대기중", "progress": 0},
        5: {"name": "클러스터 이름 제안", "status": "대기중", "progress": 0}
    }
    return phases

def update_phase_status(phases, phase_num, status, progress=None):
    """단계별 상태 업데이트"""
    phases[phase_num]["status"] = status
    if progress is not None:
        phases[phase_num]["progress"] = progress

    display_phase_status(phases)

def display_phase_status(phases):
    """Phase 진행 상황 표시"""
    clear_output(wait=True)

    print("🎯 Phase 5: 비지도 학습 (클러스터링) 진행 상황")
    print("="*60)

    status_icons = {
        "대기중": "⏳",
        "진행중": "🔄",
        "완료": "✅",
        "오류": "❌"
    }

    for phase_num, info in phases.items():
        icon = status_icons.get(info["status"], "❓")
        progress_bar = ""

        if info["progress"] > 0:
            bar_length = 20
            filled = int(bar_length * info["progress"] / 100)
            progress_bar = f" |{'█' * filled}{'░' * (bar_length - filled)}| {info['progress']}%"

        print(f"{icon} {phase_num}단계: {info['name']} - {info['status']}{progress_bar}")

    print("-"*60)

def show_clustering_progress(current_step, total_steps, step_name, sub_progress=None):
    """세부 진행도 표시"""
    main_progress = (current_step / total_steps) * 100

    progress_msg = f"🔄 {step_name} ({current_step}/{total_steps})"
    if sub_progress is not None:
        progress_msg += f" - {sub_progress}%"

    print(f"\r{progress_msg}", end='', flush=True)

    if current_step == total_steps:
        print(f"\n✅ {step_name} 완료!")

print("📊 진행도 표시 유틸리티 함수 로드 완료!")
```
### 1. 최적 클러스터 수 결정 함수들
``` python
print("🎯 Phase 5: 비지도 학습 (클러스터링) 시작!")
print("="*60)

# 1. 최적 클러스터 수 결정 함수들 (진행도 포함)

def find_optimal_clusters_elbow(X, max_clusters=15, random_state=42, phases=None, phase_num=1):
    """Elbow Method를 사용한 최적 클러스터 수 찾기 (진행도 포함)"""
    if phases:
        update_phase_status(phases, phase_num, "진행중", 0)

    print("📈 Elbow Method로 최적 클러스터 수 탐색")
    print("-" * 40)

    # 샘플링 (성능 향상을 위해)
    if len(X) > 10000:
        sample_size = 10000
        X_sample = X.sample(n=sample_size, random_state=random_state)
        print(f"🎯 성능을 위해 {sample_size:,}개 샘플 사용")
    else:
        X_sample = X

    # 클러스터 수별 WCSS (Within-Cluster Sum of Squares) 계산
    wcss = []
    k_range = range(2, max_clusters + 1)
    total_steps = len(k_range)

    print("🔄 클러스터 수별 WCSS 계산 중...")

    # tqdm을 사용한 진행도 바
    with tqdm(k_range, desc="Elbow Method", unit="cluster") as pbar:
        for i, k in enumerate(pbar):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X_sample)
            wcss.append(kmeans.inertia_)

            # 진행도 업데이트
            progress = ((i + 1) / total_steps) * 50  # 50%까지 배정
            if phases:
                update_phase_status(phases, phase_num, "진행중", progress)

            pbar.set_postfix({'K': k, 'WCSS': f'{kmeans.inertia_:,.0f}'})
            time.sleep(0.1)  # 진행도 시각화를 위한 약간의 지연

    # Elbow Point 계산 (2차 미분 방법)
    def calculate_elbow_point(wcss_values):
        if len(wcss_values) < 3:
            return len(wcss_values) + 1

        # 1차 및 2차 차분 계산
        first_diff = np.diff(wcss_values)
        second_diff = np.diff(first_diff)

        # 2차 차분이 최대인 지점을 elbow로 선택
        elbow_idx = np.argmax(second_diff) + 2  # +2는 인덱스 보정
        return elbow_idx + 2  # k는 2부터 시작하므로 +2

    print("🔄 Elbow Point 계산 중...")
    optimal_k_elbow = calculate_elbow_point(wcss)

    if phases:
        update_phase_status(phases, phase_num, "진행중", 75)

    # 시각화
    print("🎨 Elbow Curve 시각화 중...")
    plt.figure(figsize=(12, 5))

    # Elbow Curve
    plt.subplot(1, 2, 1)
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k_elbow, color='red', linestyle='--',
                label=f'Optimal K = {optimal_k_elbow}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # WCSS 감소율
    plt.subplot(1, 2, 2)
    if len(wcss) > 1:
        reduction_rate = [abs(wcss[i-1] - wcss[i]) / wcss[i-1] * 100 for i in range(1, len(wcss))]
        plt.plot(k_range[1:], reduction_rate, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS Reduction Rate (%)')
        plt.title('WCSS Reduction Rate')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if phases:
        update_phase_status(phases, phase_num, "완료", 100)

    print(f"🎯 Elbow Method 최적 클러스터 수: {optimal_k_elbow}")
    return optimal_k_elbow, wcss

def find_optimal_clusters_silhouette(X, max_clusters=15, random_state=42, phases=None, phase_num=1):
    """Silhouette Analysis를 사용한 최적 클러스터 수 찾기 (진행도 포함)"""
    print("\n📊 Silhouette Analysis로 최적 클러스터 수 탐색")
    print("-" * 40)

    # 샘플링 (성능 향상을 위해)
    if len(X) > 5000:  # Silhouette는 더 많은 계산이 필요
        sample_size = 5000
        X_sample = X.sample(n=sample_size, random_state=random_state)
        print(f"🎯 성능을 위해 {sample_size:,}개 샘플 사용")
    else:
        X_sample = X

    # 클러스터 수별 Silhouette Score 계산
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    total_steps = len(k_range)

    print("🔄 클러스터 수별 Silhouette Score 계산 중...")

    # tqdm을 사용한 진행도 바
    with tqdm(k_range, desc="Silhouette Analysis", unit="cluster") as pbar:
        for i, k in enumerate(pbar):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_sample)
            silhouette_avg = silhouette_score(X_sample, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            # 진행도 업데이트
            progress = 50 + ((i + 1) / total_steps) * 50  # 50%부터 100%까지
            if phases:
                update_phase_status(phases, phase_num, "진행중", progress)

            pbar.set_postfix({'K': k, 'Silhouette': f'{silhouette_avg:.4f}'})
            time.sleep(0.1)  # 진행도 시각화를 위한 약간의 지연

    # 최적 클러스터 수 (Silhouette Score가 최대인 지점)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)

    # 시각화
    print("🎨 Silhouette Score 시각화 중...")
    plt.figure(figsize=(12, 5))

    # Silhouette Score
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k_silhouette, color='blue', linestyle='--',
                label=f'Optimal K = {optimal_k_silhouette}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis for Optimal K')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bar plot
    plt.subplot(1, 2, 2)
    colors = ['red' if k == optimal_k_silhouette else 'skyblue' for k in k_range]
    plt.bar(k_range, silhouette_scores, color=colors, alpha=0.7)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Scores Comparison')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"🎯 Silhouette Analysis 최적 클러스터 수: {optimal_k_silhouette} (점수: {max_silhouette:.4f})")
    return optimal_k_silhouette, silhouette_scores

def determine_final_cluster_number(elbow_k, silhouette_k, silhouette_scores, wcss):
    """최종 클러스터 수 결정"""
    print(f"\n🎯 최종 클러스터 수 결정")
    print("-" * 40)

    print(f"📈 Elbow Method 추천: K = {elbow_k}")
    print(f"📊 Silhouette Analysis 추천: K = {silhouette_k}")

    # 결정 로직
    if abs(elbow_k - silhouette_k) <= 1:
        # 두 방법이 비슷한 결과를 제시하면 평균 사용
        final_k = int((elbow_k + silhouette_k) / 2)
        decision_reason = "두 방법의 결과가 유사함"
    elif silhouette_k <= len(silhouette_scores) + 1 and silhouette_scores[silhouette_k - 2] > 0.3:  # 인덱스 보정
        # Silhouette Score가 충분히 높으면 Silhouette 방법 우선
        final_k = silhouette_k
        decision_reason = f"Silhouette Score가 양호함 ({silhouette_scores[silhouette_k - 2]:.3f})"
    else:
        # 그렇지 않으면 Elbow 방법 우선
        final_k = elbow_k
        decision_reason = "Elbow Method를 우선 적용"

    # 비즈니스 로직 적용 (4-8개 클러스터가 해석하기 좋음)
    if final_k < 4:
        final_k = 4
        decision_reason += " (최소 4개 클러스터로 조정)"
    elif final_k > 8:
        final_k = 8
        decision_reason += " (최대 8개 클러스터로 조정)"

    print(f"🎯 최종 결정: K = {final_k}")
    print(f"📝 결정 근거: {decision_reason}")

    return final_k
```
### 2. K-Means 클러스터링 실행 함수들
``` python
# 2. K-Means 클러스터링 실행 함수들

def perform_kmeans_clustering(X, n_clusters, random_state=42, phases=None, phase_num=2):
    """K-Means 클러스터링 실행"""
    if phases:
        update_phase_status(phases, phase_num, "진행중", 0)

    print(f"\n🎯 K-Means 클러스터링 실행 (K = {n_clusters})")
    print("-" * 40)

    # K-Means 모델 훈련
    print("🔄 클러스터링 진행 중...")

    if phases:
        update_phase_status(phases, phase_num, "진행중", 25)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,  # 더 안정적인 결과를 위해 초기화 횟수 증가
        max_iter=1000
    )

    # 진행도 표시와 함께 실행
    start_time = time.time()

    print("🔄 K-Means 알고리즘 실행 중...")
    with tqdm(total=100, desc="K-Means Clustering", unit="%") as pbar:
        # 실행 시뮬레이션 (실제로는 fit_predict가 한번에 실행됨)
        for i in range(0, 101, 10):
            if i == 0:
                cluster_labels = kmeans.fit_predict(X)
            pbar.update(10)
            time.sleep(0.05)  # 진행도 시각화를 위한 지연

            if phases and i <= 90:
                update_phase_status(phases, phase_num, "진행중", 25 + (i * 0.75))

    execution_time = time.time() - start_time

    print(f"⏰ 클러스터링 실행 시간: {execution_time:.2f}초")

    # 클러스터링 품질 평가
    print("🔄 클러스터링 품질 평가 중...")
    silhouette_avg = silhouette_score(X, cluster_labels)
    inertia = kmeans.inertia_

    if phases:
        update_phase_status(phases, phase_num, "완료", 100)

    print(f"📊 클러스터링 품질 지표:")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Inertia (WCSS): {inertia:,.0f}")

    # 클러스터별 크기 확인
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    print(f"📈 클러스터별 크기:")
    for cluster_id, size in cluster_sizes.items():
        percentage = (size / len(cluster_labels)) * 100
        print(f"  클러스터 {cluster_id}: {size:,}개 ({percentage:.1f}%)")

    return kmeans, cluster_labels, {
        'silhouette_score': silhouette_avg,
        'inertia': inertia,
        'cluster_sizes': cluster_sizes,
        'execution_time': execution_time
    }

def analyze_cluster_characteristics(X, cluster_labels, feature_names, phases=None, phase_num=3):
    """클러스터별 특성 분석 (진행도 포함)"""
    if phases:
        update_phase_status(phases, phase_num, "진행중", 0)

    print(f"\n🔍 클러스터별 특성 분석")
    print("-" * 40)

    # 데이터프레임 생성
    print("🔄 데이터 준비 중...")
    df_analysis = X.copy()
    df_analysis['cluster'] = cluster_labels

    if phases:
        update_phase_status(phases, phase_num, "진행중", 25)

    # 클러스터별 통계 계산
    print("🔄 클러스터별 통계 계산 중...")
    cluster_stats = df_analysis.groupby('cluster').agg(['mean', 'std', 'median']).round(3)

    if phases:
        update_phase_status(phases, phase_num, "진행중", 50)

    # 각 클러스터의 주요 특징 추출
    cluster_profiles = {}
    n_clusters = len(np.unique(cluster_labels))

    print("🔄 클러스터 프로파일 생성 중...")

    with tqdm(range(n_clusters), desc="Cluster Analysis", unit="cluster") as pbar:
        for i, cluster_id in enumerate(pbar):
            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            cluster_mean = cluster_data.drop('cluster', axis=1).mean()

            # 전체 평균과 비교하여 상위/하위 특성 찾기
            overall_mean = X.mean()
            feature_ratios = cluster_mean / (overall_mean + 1e-8)  # 0으로 나누기 방지

            # 상위 5개와 하위 5개 특성
            top_features = feature_ratios.nlargest(5)
            bottom_features = feature_ratios.nsmallest(5)

            cluster_profiles[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_analysis) * 100,
                'top_features': top_features,
                'bottom_features': bottom_features,
                'mean_values': cluster_mean
            }

            pbar.set_postfix({'cluster': cluster_id, 'size': len(cluster_data)})

            # 진행도 업데이트
            progress = 50 + ((i + 1) / n_clusters) * 50
            if phases:
                update_phase_status(phases, phase_num, "진행중", progress)

    # 결과 출력
    print("\n📊 클러스터별 특성 요약:")
    for cluster_id, profile in cluster_profiles.items():
        print(f"\n🎯 클러스터 {cluster_id} 특성:")
        print(f"  크기: {profile['size']:,}개 ({profile['percentage']:.1f}%)")
        print(f"  상위 특성 (평균 대비):")
        for feature, ratio in profile['top_features'].head(3).items():
            print(f"    {feature:<20}: {ratio:.2f}배")
        print(f"  하위 특성 (평균 대비):")
        for feature, ratio in profile['bottom_features'].head(3).items():
            print(f"    {feature:<20}: {ratio:.2f}배")

    if phases:
        update_phase_status(phases, phase_num, "완료", 100)

    return cluster_profiles, cluster_stats

def create_cluster_visualization(X, cluster_labels, method='pca', phases=None, phase_num=4):
    """클러스터 시각화 (차원 축소) (진행도 포함)"""
    if phases:
        update_phase_status(phases, phase_num, "진행중", 0)

    print(f"\n🎨 클러스터 시각화 ({method.upper()})")
    print("-" * 40)

    n_clusters = len(np.unique(cluster_labels))

    if method == 'pca':
        # PCA를 사용한 2D 시각화
        print("🔄 PCA 차원 축소 중...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)

        if phases:
            update_phase_status(phases, phase_num, "진행중", 25)

        X_reduced = pca.fit_transform(X)

        explained_variance = pca.explained_variance_ratio_
        print(f"📊 PCA 설명 분산: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")
        print(f"📊 총 설명 분산: {sum(explained_variance):.3f}")

    elif method == 'tsne':
        # t-SNE를 사용한 2D 시각화
        from sklearn.manifold import TSNE

        # 샘플링 (t-SNE는 계산이 오래 걸림)
        if len(X) > 5000:
            sample_idx = np.random.choice(len(X), 5000, replace=False)
            X_sample = X.iloc[sample_idx]
            labels_sample = cluster_labels[sample_idx]
        else:
            X_sample = X
            labels_sample = cluster_labels

        if phases:
            update_phase_status(phases, phase_num, "진행중", 25)

        print("🔄 t-SNE 계산 중... (시간이 오래 걸릴 수 있습니다)")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_reduced = tsne.fit_transform(X_sample)
        cluster_labels = labels_sample

    if phases:
        update_phase_status(phases, phase_num, "진행중", 60)

    # 시각화
    print("🎨 그래프 생성 중...")
    plt.figure(figsize=(15, 5))

    # 1. 클러스터별 산점도
    plt.subplot(1, 3, 1)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)

    plt.title(f'Clusters Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    if phases:
        update_phase_status(phases, phase_num, "진행중", 75)

    # 2. 클러스터 중심점과 함께 표시
    plt.subplot(1, 3, 2)
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)

        # 클러스터 중심점 계산 및 표시
        if np.sum(mask) > 0:  # 클러스터에 데이터가 있는 경우만
            center_x = np.mean(X_reduced[mask, 0])
            center_y = np.mean(X_reduced[mask, 1])
            plt.scatter(center_x, center_y, c='black', marker='x', s=200, linewidths=3)
            plt.annotate(f'C{i}', (center_x, center_y), xytext=(5, 5),
                        textcoords='offset points', fontweight='bold')

    plt.title(f'Clusters with Centroids ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)

    # 3. 클러스터 크기 비교
    plt.subplot(1, 3, 3)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.pie(counts, labels=[f'Cluster {i}' for i in unique], autopct='%1.1f%%',
            colors=colors[:len(unique)])
    plt.title('Cluster Size Distribution')

    plt.tight_layout()
    plt.show()

    if phases:
        update_phase_status(phases, phase_num, "완료", 100)

    return X_reduced

def suggest_cluster_names(cluster_profiles, phases=None, phase_num=5):
    """클러스터 특성을 바탕으로 의미있는 이름 제안 (진행도 포함)"""
    if phases:
        update_phase_status(phases, phase_num, "진행중", 0)

    print(f"\n🏷️ 클러스터 의미 분석 및 이름 제안")
    print("-" * 40)

    cluster_names = {}
    total_clusters = len(cluster_profiles)

    # 특성 기반 이름 제안 로직
    feature_keywords = {
        'kill': ['Aggressive', 'Fighter', 'Killer'],
        'damage': ['Damager', 'Combatant', 'Attacker'],
        'heal': ['Survivor', 'Medic', 'Cautious'],
        'distance': ['Explorer', 'Wanderer', 'Mobile'],
        'boost': ['Strategic', 'Prepared', 'Tactical'],
        'weapon': ['Collector', 'Arsenal', 'Armed'],
        'assist': ['Supporter', 'Team Player', 'Helper'],
        'efficiency': ['Efficient', 'Optimal', 'Skilled']
    }

    print("🔄 클러스터 이름 생성 중...")

    with tqdm(cluster_profiles.items(), desc="Naming Clusters", unit="cluster") as pbar:
        for i, (cluster_id, profile) in enumerate(pbar):
            top_features = profile['top_features'].head(3)

            # 상위 특성을 바탕으로 키워드 점수 계산
            keyword_scores = {}
            for keyword_group, names in feature_keywords.items():
                score = 0
                for feature_name in top_features.index:
                    if keyword_group in feature_name.lower():
                        score += top_features[feature_name]

                if score > 0:
                    for name in names:
                        keyword_scores[name] = keyword_scores.get(name, 0) + score

            # 가장 높은 점수의 키워드 선택
            if keyword_scores:
                suggested_name = max(keyword_scores.items(), key=lambda x: x[1])[0]
            else:
                suggested_name = f"Type_{cluster_id}"

            cluster_names[cluster_id] = suggested_name

            pbar.set_postfix({'cluster': cluster_id, 'name': suggested_name})

            # 진행도 업데이트
            progress = ((i + 1) / total_clusters) * 100
            if phases:
                update_phase_status(phases, phase_num, "진행중", progress)

    print("\n🎯 클러스터 이름 제안 결과:")
    for cluster_id, name in cluster_names.items():
        profile = cluster_profiles[cluster_id]
        top_features = profile['top_features'].head(2)
        print(f"🎯 클러스터 {cluster_id}: '{name}' 플레이어")
        print(f"  주요 특징: {', '.join(top_features.index)}")
        print(f"  크기: {profile['size']:,}명 ({profile['percentage']:.1f}%)")

    if phases:
        update_phase_status(phases, phase_num, "완료", 100)

    return cluster_names
```
### 3. 종합 클러스터링 파이프라인
``` python
# 3. 종합 클러스터링 파이프라인

def comprehensive_clustering_pipeline(X, feature_names, max_clusters=12):
    """종합적인 클러스터링 분석 파이프라인 (진행도 포함)"""
    print("🚀 종합 클러스터링 분석 파이프라인 시작")
    print("="*60)

    # 전체 진행도 추적기 초기화
    phases = create_phase_tracker()
    display_phase_status(phases)

    start_time = time.time()
    clustering_results = {}

    # 1. 최적 클러스터 수 탐색
    print(f"\n{'='*20} 1단계: 최적 클러스터 수 탐색 {'='*20}")
    update_phase_status(phases, 1, "진행중", 0)

    optimal_k_elbow, wcss = find_optimal_clusters_elbow(X, max_clusters, phases=phases, phase_num=1)
    optimal_k_silhouette, silhouette_scores = find_optimal_clusters_silhouette(X, max_clusters, phases=phases, phase_num=1)

    # 최종 클러스터 수 결정
    final_k = determine_final_cluster_number(optimal_k_elbow, optimal_k_silhouette,
                                           silhouette_scores, wcss)

    clustering_results['optimal_clusters'] = {
        'elbow_method': optimal_k_elbow,
        'silhouette_method': optimal_k_silhouette,
        'final_decision': final_k,
        'wcss_values': wcss,
        'silhouette_scores': silhouette_scores
    }

    # 2. K-Means 클러스터링 실행
    print(f"\n{'='*20} 2단계: K-Means 클러스터링 실행 {'='*20}")
    kmeans_model, cluster_labels, clustering_metrics = perform_kmeans_clustering(
        X, final_k, phases=phases, phase_num=2)

    clustering_results['kmeans_model'] = kmeans_model
    clustering_results['cluster_labels'] = cluster_labels
    clustering_results['clustering_metrics'] = clustering_metrics

    # 3. 클러스터 특성 분석
    print(f"\n{'='*20} 3단계: 클러스터 특성 분석 {'='*20}")
    cluster_profiles, cluster_stats = analyze_cluster_characteristics(
        X, cluster_labels, feature_names, phases=phases, phase_num=3)

    clustering_results['cluster_profiles'] = cluster_profiles
    clustering_results['cluster_stats'] = cluster_stats

    # 4. 클러스터 시각화
    print(f"\n{'='*20} 4단계: 클러스터 시각화 {'='*20}")
    X_reduced_pca = create_cluster_visualization(
        X, cluster_labels, method='pca', phases=phases, phase_num=4)

    clustering_results['visualization_data'] = {
        'pca_components': X_reduced_pca,
        'cluster_labels': cluster_labels
    }

    # 5. 클러스터 이름 제안
    print(f"\n{'='*20} 5단계: 클러스터 이름 제안 {'='*20}")
    cluster_names = suggest_cluster_names(cluster_profiles, phases=phases, phase_num=5)

    clustering_results['cluster_names'] = cluster_names

    # 실행 시간 계산
    execution_time = time.time() - start_time

    # 최종 완료 상태 표시
    for phase_num in phases.keys():
        update_phase_status(phases, phase_num, "완료", 100)

    time.sleep(1)  # 최종 상태 확인을 위한 잠시 대기

    # 최종 요약
    print(f"\n{'='*20} 🎉 클러스터링 완료 요약 🎉 {'='*20}")
    print(f"⏰ 총 실행 시간: {execution_time:.1f}초")
    print(f"🎯 최종 클러스터 수: {final_k}개")
    print(f"📊 Silhouette Score: {clustering_metrics['silhouette_score']:.4f}")
    print(f"🏷️ 발견된 플레이어 유형:")

    for cluster_id, name in cluster_names.items():
        size = cluster_profiles[cluster_id]['size']
        pct = cluster_profiles[cluster_id]['percentage']
        print(f"  🎮 클러스터 {cluster_id}: {name} ({size:,}명, {pct:.1f}%)")

    print(f"✅ Phase 5 완료! Phase 6 (딥러닝 분류)로 진행 가능")

    return clustering_results

def run_phase5_pipeline(modeling_data):
    """Phase 5 전체 파이프라인 실행 (진행도 포함)"""
    print("🎮 PUBG Phase 5: 비지도 학습 (클러스터링) 시작!")
    print("="*60)

    # 메모리 확인
    check_memory_usage()

    # 모델링 데이터에서 특성 추출 (타겟 제외)
    X_train = modeling_data['X_train']
    feature_names = modeling_data['feature_names']

    print(f"📊 클러스터링 데이터: {X_train.shape[0]:,}행 x {X_train.shape[1]}열")

    # 전체 파이프라인 시작 알림
    print("\n🚀 5단계 클러스터링 파이프라인을 시작합니다...")
    print("📋 예상 소요 시간: 5-10분")

    # 종합 클러스터링 실행
    clustering_results = comprehensive_clustering_pipeline(X_train, feature_names)

    return clustering_results
```
### 4. 실행 가이드
``` python
# 4. 실행 가이드

def display_phase5_guide():
    """Phase 5 실행 가이드 표시"""
    print("\n📋 Phase 5 실행 방법:")
    print("="*50)
    print("# 1. Phase 4에서 준비된 모델링 데이터 사용")
    print("clustering_results = run_phase5_pipeline(modeling_data)")
    print()
    print("# 2. 결과 확인")
    print("print('클러스터 수:', clustering_results['optimal_clusters']['final_decision'])")
    print("print('클러스터 이름:', clustering_results['cluster_names'])")
    print("print('품질 점수:', clustering_results['clustering_metrics']['silhouette_score'])")
    print()
    print("# 3. 클러스터 레이블 추출")
    print("cluster_labels = clustering_results['cluster_labels']")
    print("cluster_profiles = clustering_results['cluster_profiles']")

# 가이드 표시
display_phase5_guide()

print("\n🎯 Phase 5 준비 완료!")
print("🧠 다음 단계: Phase 6 - 지도 학습 (딥러닝 분류 모델)")
```

#### 4. 실행 가이드 결과

``` bash
📋 Phase 5 실행 방법:
==================================================
# 1. Phase 4에서 준비된 모델링 데이터 사용
clustering_results = run_phase5_pipeline(modeling_data)

# 2. 결과 확인
print('클러스터 수:', clustering_results['optimal_clusters']['final_decision'])
print('클러스터 이름:', clustering_results['cluster_names'])
print('품질 점수:', clustering_results['clustering_metrics']['silhouette_score'])

# 3. 클러스터 레이블 추출
cluster_labels = clustering_results['cluster_labels']
cluster_profiles = clustering_results['cluster_profiles']

🎯 Phase 5 준비 완료!
🧠 다음 단계: Phase 6 - 지도 학습 (딥러닝 분류 모델)
```

### 5. 추가 유틸리티 함수들
``` python
# 5. 추가 유틸리티 함수들

def save_clustering_model(clustering_results, filename='pubg_clustering_model.pkl'):
    """클러스터링 모델 저장 (진행도 포함)"""
    import pickle

    print("💾 클러스터링 모델 저장 중...")

    # 모델과 주요 결과만 저장 (메모리 효율성)
    save_data = {
        'kmeans_model': clustering_results['kmeans_model'],
        'cluster_names': clustering_results['cluster_names'],
        'cluster_profiles': clustering_results['cluster_profiles'],
        'optimal_clusters': clustering_results['optimal_clusters'],
        'clustering_metrics': clustering_results['clustering_metrics']
    }

    try:
        with tqdm(total=100, desc="Saving Model", unit="%") as pbar:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
                pbar.update(100)

        print(f"✅ 클러스터링 모델 저장 완료: {filename}")
        return True
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")
        return False

def predict_player_cluster(kmeans_model, player_features, cluster_names=None):
    """신규 플레이어의 클러스터 예측 (진행도 포함)"""
    print("🔮 플레이어 클러스터 예측 중...")

    with tqdm(total=100, desc="Predicting", unit="%") as pbar:
        pbar.update(50)
        cluster_id = kmeans_model.predict([player_features])[0]
        pbar.update(50)

    if cluster_names and cluster_id in cluster_names:
        cluster_name = cluster_names[cluster_id]
        print(f"🎯 예측 결과: 클러스터 {cluster_id} ({cluster_name})")
    else:
        print(f"🎯 예측 결과: 클러스터 {cluster_id}")

    return cluster_id

def generate_clustering_report(clustering_results):
    """클러스터링 결과 보고서 생성"""
    print("📄 클러스터링 분석 보고서 생성 중...")

    with tqdm(total=100, desc="Generating Report", unit="%") as pbar:
        # 보고서 템플릿
        report = []
        report.append("=" * 60)
        report.append("🎮 PUBG 플레이어 클러스터링 분석 보고서")
        report.append("=" * 60)

        pbar.update(20)

        # 기본 정보
        final_k = clustering_results['optimal_clusters']['final_decision']
        silhouette_score = clustering_results['clustering_metrics']['silhouette_score']

        report.append(f"\n📊 분석 요약:")
        report.append(f"• 최종 클러스터 수: {final_k}개")
        report.append(f"• 클러스터링 품질 (Silhouette Score): {silhouette_score:.4f}")

        pbar.update(30)

        # 클러스터별 상세 정보
        report.append(f"\n🎯 발견된 플레이어 유형:")

        for cluster_id, name in clustering_results['cluster_names'].items():
            profile = clustering_results['cluster_profiles'][cluster_id]
            report.append(f"\n🎮 클러스터 {cluster_id}: {name}")
            report.append(f"   크기: {profile['size']:,}명 ({profile['percentage']:.1f}%)")

            top_features = profile['top_features'].head(3)
            report.append(f"   주요 특성:")
            for feature, ratio in top_features.items():
                report.append(f"   • {feature}: 평균 대비 {ratio:.2f}배")

        pbar.update(30)

        # 추천사항
        report.append(f"\n💡 비즈니스 인사이트:")
        report.append("• 각 플레이어 유형별 맞춤형 컨텐츠 제공 가능")
        report.append("• 플레이어 행동 패턴 기반 게임 밸런싱 개선")
        report.append("• 신규 플레이어 온보딩 전략 최적화")

        pbar.update(20)

        report_text = "\n".join(report)

        # 파일 저장
        try:
            with open("pubg_clustering_report.txt", "w", encoding="utf-8") as f:
                f.write(report_text)
            print("✅ 보고서 파일 저장 완료: pubg_clustering_report.txt")
        except:
            pass

        print("\n" + report_text)

        return report_text
```
### 실행
``` python
# 진행도가 포함된 종합 클러스터링 분석 실행
print("🚀 진행도 표시와 함께 클러스터링 시작!")

# 대시보드 생성 (선택사항)
# create_progress_dashboard() # Removed the call to the undefined function

# 종합 클러스터링 분석 (모든 진행도 표시 포함)
final_results = run_phase5_pipeline(modeling_data)

# 핵심 결과 확인
print("\n🎯 최종 결과 요약:")
print('📊 클러스터 수:', final_results['optimal_clusters']['final_decision'])
print('📈 품질 평가:', final_results['clustering_metrics']['silhouette_score'])
print('🏷️ 클러스터 유형:')
for cluster_id, name in final_results['cluster_names'].items():
    size = final_results['cluster_profiles'][cluster_id]['size']
    pct = final_results['cluster_profiles'][cluster_id]['percentage']
    print(f'   🎮 클러스터 {cluster_id}: {name} ({size:,}명, {pct:.1f}%)')

# 보고서 생성 (선택사항)
print("\n📄 상세 분석 보고서 생성...")
clustering_report = generate_clustering_report(final_results)

# 모델 저장 (선택사항)
print("\n💾 클러스터링 모델 저장...")
save_success = save_clustering_model(final_results)

print("\n🎉 Phase 5 클러스터링 (진행도 포함) 완료!")
print("✅ 다음 단계: Phase 6 - 딥러닝 분류 모델 개발")
```

#### 실행 결과

``` bash
🎯 Phase 5: 비지도 학습 (클러스터링) 진행 상황
============================================================
✅ 1단계: 최적 클러스터 수 탐색 - 완료 |████████████████████| 100%
✅ 2단계: K-Means 클러스터링 실행 - 완료 |████████████████████| 100%
✅ 3단계: 클러스터 특성 분석 - 완료 |████████████████████| 100%
✅ 4단계: 클러스터 시각화 - 완료 |████████████████████| 100%
✅ 5단계: 클러스터 이름 제안 - 완료 |████████████████████| 100%
------------------------------------------------------------

==================== 🎉 클러스터링 완료 요약 🎉 ====================
⏰ 총 실행 시간: 104.5초
🎯 최종 클러스터 수: 8개
📊 Silhouette Score: 0.1391
🏷️ 발견된 플레이어 유형:
  🎮 클러스터 0: Survivor (14,527명, 18.2%)
  🎮 클러스터 1: Survivor (24,981명, 31.2%)
  🎮 클러스터 2: Explorer (10,756명, 13.4%)
  🎮 클러스터 3: Explorer (15,898명, 19.9%)
  🎮 클러스터 4: Explorer (4,312명, 5.4%)
  🎮 클러스터 5: Explorer (4,046명, 5.1%)
  🎮 클러스터 6: Explorer (5,391명, 6.7%)
  🎮 클러스터 7: Aggressive (89명, 0.1%)
✅ Phase 5 완료! Phase 6 (딥러닝 분류)로 진행 가능

🎯 최종 결과 요약:
📊 클러스터 수: 8
📈 품질 평가: 0.1391231221671035
🏷️ 클러스터 유형:
   🎮 클러스터 0: Survivor (14,527명, 18.2%)
   🎮 클러스터 1: Survivor (24,981명, 31.2%)
   🎮 클러스터 2: Explorer (10,756명, 13.4%)
   🎮 클러스터 3: Explorer (15,898명, 19.9%)
   🎮 클러스터 4: Explorer (4,312명, 5.4%)
   🎮 클러스터 5: Explorer (4,046명, 5.1%)
   🎮 클러스터 6: Explorer (5,391명, 6.7%)
   🎮 클러스터 7: Aggressive (89명, 0.1%)

📄 상세 분석 보고서 생성...
📄 클러스터링 분석 보고서 생성 중...
Generating Report: 100%
 100/100 [00:00<00:00, 6477.39%/s]
✅ 보고서 파일 저장 완료: pubg_clustering_report.txt

============================================================
🎮 PUBG 플레이어 클러스터링 분석 보고서
============================================================

📊 분석 요약:
• 최종 클러스터 수: 8개
• 클러스터링 품질 (Silhouette Score): 0.1391

🎯 발견된 플레이어 유형:

🎮 클러스터 0: Survivor
   크기: 14,527명 (18.2%)
   주요 특성:
   • heal_boost_ratio: 평균 대비 775.29배
   • assists: 평균 대비 479.86배
   • has_swimDistance: 평균 대비 294.02배

🎮 클러스터 1: Survivor
   크기: 24,981명 (31.2%)
   주요 특성:
   • heal_boost_ratio: 평균 대비 1861.49배
   • assists: 평균 대비 964.62배
   • damage_per_kill: 평균 대비 864.65배

🎮 클러스터 2: Explorer
   크기: 10,756명 (13.4%)
   주요 특성:
   • walkDistance_log: 평균 대비 3743.08배
   • walkDistance: 평균 대비 1179.09배
   • revives: 평균 대비 626.25배

🎮 클러스터 3: Explorer
   크기: 15,898명 (19.9%)
   주요 특성:
   • walkDistance_log: 평균 대비 2245.80배
   • longestKill: 평균 대비 610.84배
   • has_kills: 평균 대비 501.94배

🎮 클러스터 4: Explorer
   크기: 4,312명 (5.4%)
   주요 특성:
   • walkDistance_log: 평균 대비 4451.52배
   • walkDistance: 평균 대비 1845.04배
   • revives: 평균 대비 1551.02배

🎮 클러스터 5: Explorer
   크기: 4,046명 (5.1%)
   주요 특성:
   • walkDistance_log: 평균 대비 4139.77배
   • walkDistance: 평균 대비 1544.91배
   • weaponsAcquired: 평균 대비 451.79배

🎮 클러스터 6: Explorer
   크기: 5,391명 (6.7%)
   주요 특성:
   • walkDistance_log: 평균 대비 3995.30배
   • matchDuration: 평균 대비 1400.69배
   • walkDistance: 평균 대비 1327.73배

🎮 클러스터 7: Aggressive
   크기: 89명 (0.1%)
   주요 특성:
   • kill_efficiency: 평균 대비 23396.88배
   • damage_per_kill: 평균 대비 1435.73배
   • assists: 평균 대비 920.03배

💡 비즈니스 인사이트:
• 각 플레이어 유형별 맞춤형 컨텐츠 제공 가능
• 플레이어 행동 패턴 기반 게임 밸런싱 개선
• 신규 플레이어 온보딩 전략 최적화

💾 클러스터링 모델 저장...
💾 클러스터링 모델 저장 중...
Saving Model: 100%
 100/100 [00:00<00:00, 4755.55%/s]
✅ 클러스터링 모델 저장 완료: pubg_clustering_model.pkl

🎉 Phase 5 클러스터링 (진행도 포함) 완료!
✅ 다음 단계: Phase 6 - 딥러닝 분류 모델 개발
```

## Phase 6: 지도 학습 (딥러닝 분류 모델)

### 1. 데이터 준비 및 전처리 함수들
``` python

print("🧠 Phase 6: 지도 학습 (딥러닝 분류 모델) 시작!")
print("="*60)

# 1. 데이터 준비 및 전처리 함수들

def prepare_classification_data(modeling_data, clustering_results):
    """분류를 위한 데이터 준비"""
    print("🎯 분류 모델용 데이터 준비")
    print("-" * 40)

    # 클러스터 레이블을 타겟으로 사용
    X_train = modeling_data['X_train']
    X_test = modeling_data['X_test']

    # 클러스터 레이블 (Phase 5에서 생성된 것)
    cluster_labels = clustering_results['cluster_labels']
    n_clusters = len(np.unique(cluster_labels))

    # 훈련 데이터의 클러스터 레이블 사용
    y_train_clusters = cluster_labels

    # 테스트 데이터에 대해서도 클러스터 예측 (K-Means 모델 사용)
    kmeans_model = clustering_results['kmeans_model']
    y_test_clusters = kmeans_model.predict(X_test)

    print(f"📊 분류 데이터 정보:")
    print(f"  훈련 특성: {X_train.shape}")
    print(f"  테스트 특성: {X_test.shape}")
    print(f"  클러스터 수: {n_clusters}개")
    print(f"  클러스터 분포 (훈련):")

    unique, counts = np.unique(y_train_clusters, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        percentage = count / len(y_train_clusters) * 100
        cluster_name = clustering_results['cluster_names'].get(cluster_id, f'Cluster_{cluster_id}')
        print(f"    {cluster_name}: {count:,}개 ({percentage:.1f}%)")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_clusters,
        'y_test': y_test_clusters,
        'n_classes': n_clusters,
        'feature_names': modeling_data['feature_names'],
        'cluster_names': clustering_results['cluster_names']
    }

def create_validation_split(X_train, y_train, validation_split=0.2, random_state=42):
    """검증용 데이터 분할"""
    print(f"🔀 검증 데이터 분할 ({validation_split*100:.0f}%)")
    print("-" * 40)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_train  # 클러스터 비율 유지
    )

    print(f"📊 분할 결과:")
    print(f"  훈련 세트: {X_train_split.shape[0]:,}개")
    print(f"  검증 세트: {X_val_split.shape[0]:,}개")

    return X_train_split, X_val_split, y_train_split, y_val_split
```
### 2. 딥러닝 모델 아키텍처 함수들
``` python
# 2. 딥러닝 모델 아키텍처 함수들

def create_basic_neural_network(input_dim, n_classes, hidden_units=[128, 64, 32]):
    """기본 신경망 모델 생성"""
    print(f"🧠 기본 신경망 모델 구축")
    print(f"  입력 차원: {input_dim}")
    print(f"  출력 클래스: {n_classes}")
    print(f"  은닉층: {hidden_units}")
    print("-" * 40)

    model = keras.Sequential([
        # 입력층
        layers.Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # 첫 번째 은닉층
        layers.Dense(hidden_units[1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # 두 번째 은닉층
        layers.Dense(hidden_units[2], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # 출력층
        layers.Dense(n_classes, activation='softmax')
    ])

    return model

def create_advanced_neural_network(input_dim, n_classes):
    """고급 신경망 모델 생성 (더 복잡한 아키텍처)"""
    print(f"🚀 고급 신경망 모델 구축")
    print(f"  입력 차원: {input_dim}")
    print(f"  출력 클래스: {n_classes}")
    print("-" * 40)

    # 함수형 API 사용
    inputs = layers.Input(shape=(input_dim,))

    # 첫 번째 브랜치
    x1 = layers.Dense(256, activation='relu')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.4)(x1)

    x1 = layers.Dense(128, activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)

    # 두 번째 브랜치
    x2 = layers.Dense(128, activation='relu')(inputs)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.3)(x2)

    x2 = layers.Dense(64, activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    # 브랜치 결합
    combined = layers.concatenate([x1, x2])

    # 최종 레이어들
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    # 출력층
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_residual_network(input_dim, n_classes):
    """Residual Connection을 포함한 신경망"""
    print(f"🔗 Residual 신경망 모델 구축")
    print("-" * 40)

    inputs = layers.Input(shape=(input_dim,))

    # 첫 번째 블록
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Residual 블록 1
    residual = x
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # Skip connection
    x = layers.Activation('relu')(x)

    # Residual 블록 2
    residual = layers.Dense(64)(x)  # 차원 맞춤
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)

    # 출력층
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
```
### 3. 모델 훈련 및 평가 함수들
``` python
# 3. 모델 훈련 및 평가 함수들

def compile_and_train_model(model, X_train, y_train, X_val, y_val,
                           epochs=100, batch_size=256, learning_rate=0.001):
    """모델 컴파일 및 훈련"""
    print(f"🎯 모델 훈련 시작")
    print(f"  에포크: {epochs}")
    print(f"  배치 크기: {batch_size}")
    print(f"  학습률: {learning_rate}")
    print("-" * 40)

    # 모델 컴파일
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 모델 요약 출력
    print("🏗️ 모델 아키텍처:")
    model.summary()

    # 훈련 시작
    print("\n🔄 모델 훈련 진행 중...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"⏰ 훈련 완료 시간: {training_time:.1f}초")

    return model, history

def evaluate_model_performance(model, X_test, y_test, cluster_names):
    """모델 성능 평가"""
    print(f"\n📊 모델 성능 평가")
    print("-" * 40)

    # 예측 수행
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 기본 성능 지표
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 클래스별 성능 리포트
    class_names = [cluster_names.get(i, f'Cluster_{i}') for i in range(len(cluster_names))]
    classification_rep = classification_report(y_test, y_pred,
                                             target_names=class_names,
                                             output_dict=True)

    print(f"\n📈 클래스별 성능:")
    for class_name in class_names:
        if class_name in classification_rep:
            metrics = classification_rep[class_name]
            print(f"  {class_name:<15}: "
                  f"정밀도={metrics['precision']:.3f}, "
                  f"재현율={metrics['recall']:.3f}, "
                  f"F1={metrics['f1-score']:.3f}")

    # 전체 평균
    macro_avg = classification_rep['macro avg']
    weighted_avg = classification_rep['weighted avg']

    print(f"\n📊 전체 평균:")
    print(f"  Macro Avg    : F1={macro_avg['f1-score']:.3f}")
    print(f"  Weighted Avg : F1={weighted_avg['f1-score']:.3f}")

    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': classification_rep
    }

def create_confusion_matrix_plot(y_test, y_pred, cluster_names):
    """혼동 행렬 시각화"""
    print(f"\n🔄 혼동 행렬 생성")
    print("-" * 40)

    from sklearn.metrics import confusion_matrix

    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, y_pred)

    # 클래스 이름
    class_names = [cluster_names.get(i, f'C{i}') for i in range(len(cluster_names))]

    # 시각화
    plt.figure(figsize=(12, 10))

    # 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 히트맵 생성
    sns.heatmap(cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True)

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 분류 정확도 (대각선 성분)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    print(f"📊 클래스별 분류 정확도:")
    for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
        print(f"  {class_name:<15}: {acc:.3f} ({acc*100:.1f}%)")

    return cm, cm_normalized

def plot_training_history(history):
    """훈련 과정 시각화"""
    print(f"\n📈 훈련 과정 시각화")
    print("-" * 40)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 손실 함수
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 정확도
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 최종 성능 요약
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print(f"📊 최종 훈련 성능:")
    print(f"  훈련 정확도: {final_train_acc:.4f}")
    print(f"  검증 정확도: {final_val_acc:.4f}")
    print(f"  훈련 손실: {final_train_loss:.4f}")
    print(f"  검증 손실: {final_val_loss:.4f}")

    # 과적합 여부 확인
    overfitting_gap = final_train_acc - final_val_acc
    if overfitting_gap > 0.1:
        print(f"⚠️ 과적합 의심 (차이: {overfitting_gap:.3f})")
    else:
        print(f"✅ 적절한 일반화 성능 (차이: {overfitting_gap:.3f})")
```
### 4. 모델 해석 및 특성 중요도 분석
``` python
# 4. 모델 해석 및 특성 중요도 분석

def analyze_feature_importance_with_permutation(model, X_test, y_test, feature_names):
    """Permutation Importance를 사용한 특성 중요도 분석"""
    print(f"\n🔍 Permutation Importance 특성 중요도 분석")
    print("-" * 40)

    from sklearn.inspection import permutation_importance

    # 베이스라인 성능
    baseline_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

    # 각 특성을 하나씩 셔플하여 성능 변화 측정
    feature_importance_scores = []

    print("🔄 특성별 중요도 계산 중...")
    for i, feature_name in enumerate(feature_names):
        if i % 10 == 0:  # 진행상황 표시
            print(f"  진행: {i+1}/{len(feature_names)}")

        # 특성 복사 및 셔플
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled.iloc[:, i].values)

        # 셔플된 데이터로 성능 측정
        shuffled_accuracy = model.evaluate(X_test_shuffled, y_test, verbose=0)[1]

        # 중요도 = 베이스라인 성능 - 셔플된 성능
        importance = baseline_accuracy - shuffled_accuracy
        feature_importance_scores.append(importance)

    # 결과 정리
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_scores
    }).sort_values('importance', ascending=False)

    # 상위 15개 특성 시각화
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)

    bars = plt.barh(range(len(top_features)), top_features['importance'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))

    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Accuracy Drop)')
    plt.title('Top 15 Feature Importance (Permutation Method)',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # 값 표시
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"📊 상위 10개 중요 특성:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

    return feature_importance_df

def create_prediction_confidence_analysis(y_pred_proba, y_test, cluster_names):
    """예측 신뢰도 분석"""
    print(f"\n🎯 예측 신뢰도 분석")
    print("-" * 40)

    # 최대 확률 (신뢰도)
    max_probabilities = np.max(y_pred_proba, axis=1)

    # 신뢰도 구간별 정확도 분석
    confidence_bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    bin_accuracies = []
    bin_counts = []

    for i in range(len(confidence_bins)-1):
        lower_bound = confidence_bins[i]
        upper_bound = confidence_bins[i+1]

        # 해당 신뢰도 구간에 속하는 예측들
        mask = (max_probabilities >= lower_bound) & (max_probabilities < upper_bound)

        if np.sum(mask) > 0:
            bin_predictions = np.argmax(y_pred_proba[mask], axis=1)
            bin_true_labels = y_test[mask]
            bin_accuracy = np.mean(bin_predictions == bin_true_labels)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 신뢰도 분포
    axes[0].hist(max_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(np.mean(max_probabilities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(max_probabilities):.3f}')
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Prediction Confidence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 신뢰도별 정확도
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7,
               color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Confidence Interval')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Confidence Level')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 통계 요약
    print(f"📊 신뢰도 통계:")
    print(f"  평균 신뢰도: {np.mean(max_probabilities):.3f}")
    print(f"  신뢰도 표준편차: {np.std(max_probabilities):.3f}")
    print(f"  높은 신뢰도 (>0.8) 비율: {np.mean(max_probabilities > 0.8)*100:.1f}%")
    print(f"  낮은 신뢰도 (<0.5) 비율: {np.mean(max_probabilities < 0.5)*100:.1f}%")

    return max_probabilities, bin_accuracies
```
### 5. 다중 모델 비교 및 앙상블
``` python
# 5. 다중 모델 비교 및 앙상블

def train_multiple_models(classification_data):
    """여러 모델 훈련 및 비교"""
    print(f"\n🏆 다중 모델 비교 분석")
    print("="*50)

    X_train = classification_data['X_train']
    y_train = classification_data['y_train']
    n_classes = classification_data['n_classes']
    input_dim = X_train.shape[1]

    # 검증 데이터 분할
    X_train_split, X_val_split, y_train_split, y_val_split = create_validation_split(
        X_train, y_train, validation_split=0.2
    )

    models = {}
    histories = {}

    # 1. 기본 모델
    print(f"\n{'='*20} 기본 신경망 모델 훈련 {'='*20}")
    basic_model = create_basic_neural_network(input_dim, n_classes)
    basic_model, basic_history = compile_and_train_model(
        basic_model, X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=50, batch_size=256
    )
    models['Basic'] = basic_model
    histories['Basic'] = basic_history

    # 2. 고급 모델
    print(f"\n{'='*20} 고급 신경망 모델 훈련 {'='*20}")
    advanced_model = create_advanced_neural_network(input_dim, n_classes)
    advanced_model, advanced_history = compile_and_train_model(
        advanced_model, X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=50, batch_size=256
    )
    models['Advanced'] = advanced_model
    histories['Advanced'] = advanced_history

    # 3. Residual 모델
    print(f"\n{'='*20} Residual 신경망 모델 훈련 {'='*20}")
    residual_model = create_residual_network(input_dim, n_classes)
    residual_model, residual_history = compile_and_train_model(
        residual_model, X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=50, batch_size=256
    )
    models['Residual'] = residual_model
    histories['Residual'] = residual_history

    return models, histories, (X_val_split, y_val_split)

def compare_model_performances(models, X_test, y_test, cluster_names):
    """모델 성능 비교"""
    print(f"\n📊 모델 성능 비교")
    print("="*50)

    model_performances = {}

    for model_name, model in models.items():
        print(f"\n🔍 {model_name} 모델 평가:")
        performance = evaluate_model_performance(model, X_test, y_test, cluster_names)
        model_performances[model_name] = performance

        print(f"  정확도: {performance['accuracy']:.4f}")
        print(f"  F1 (Macro): {performance['classification_report']['macro avg']['f1-score']:.4f}")

    # 성능 비교 시각화
    model_names = list(model_performances.keys())
    accuracies = [model_performances[name]['accuracy'] for name in model_names]
    f1_scores = [model_performances[name]['classification_report']['macro avg']['f1-score']
                 for name in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 정확도 비교
    bars1 = axes[0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)

    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # F1 점수 비교
    bars2 = axes[1].bar(model_names, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1-Score (Macro Average)')
    axes[1].set_ylim(0, 1)

    for bar, f1 in zip(bars2, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 최고 성능 모델 선택
    best_model_name = max(model_performances.keys(),
                         key=lambda x: model_performances[x]['accuracy'])

    print(f"\n🏆 최고 성능 모델: {best_model_name}")
    print(f"  정확도: {model_performances[best_model_name]['accuracy']:.4f}")

    return model_performances, best_model_name

def create_ensemble_model(models, X_test, y_test):
    """앙상블 모델 생성 (소프트 보팅)"""
    print(f"\n🤝 앙상블 모델 생성 (소프트 보팅)")
    print("-" * 40)

    # 각 모델의 예측 확률 수집
    model_predictions = []
    for model_name, model in models.items():
        pred_proba = model.predict(X_test, verbose=0)
        model_predictions.append(pred_proba)
        print(f"✅ {model_name} 모델 예측 완료")

    # 평균 앙상블
    ensemble_proba = np.mean(model_predictions, axis=0)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    # 앙상블 성능 평가
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    print(f"📊 앙상블 모델 성능:")
    print(f"  정확도: {ensemble_accuracy:.4f}")

    # 개별 모델과 비교
    print(f"\n📈 성능 향상 비교:")
    for model_name, model in models.items():
        individual_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        individual_acc = accuracy_score(y_test, individual_pred)
        improvement = ensemble_accuracy - individual_acc
        print(f"  vs {model_name}: {improvement:+.4f} ({improvement*100:+.2f}%)")

    return ensemble_pred, ensemble_proba, ensemble_accuracy
```
### 6. 종합 딥러닝 파이프라인
``` python
# 6. 종합 딥러닝 파이프라인

def comprehensive_deep_learning_pipeline(modeling_data, clustering_results):
    """종합적인 딥러닝 분류 파이프라인"""
    print("🚀 종합 딥러닝 분류 파이프라인 시작")
    print("="*60)

    start_time = time.time()
    dl_results = {}

    # 1. 분류 데이터 준비
    print(f"\n{'='*20} 1단계: 분류 데이터 준비 {'='*20}")
    classification_data = prepare_classification_data(modeling_data, clustering_results)
    dl_results['classification_data'] = classification_data

    # 2. 다중 모델 훈련
    print(f"\n{'='*20} 2단계: 다중 모델 훈련 {'='*20}")
    models, histories, validation_data = train_multiple_models(classification_data)
    dl_results['models'] = models
    dl_results['histories'] = histories

    # 3. 훈련 과정 시각화
    print(f"\n{'='*20} 3단계: 훈련 과정 시각화 {'='*20}")
    for model_name, history in histories.items():
        print(f"\n📈 {model_name} 모델 훈련 과정:")
        plot_training_history(history)

    # 4. 모델 성능 평가 및 비교
    print(f"\n{'='*20} 4단계: 모델 성능 평가 {'='*20}")
    X_test = classification_data['X_test']
    y_test = classification_data['y_test']
    cluster_names = classification_data['cluster_names']

    model_performances, best_model_name = compare_model_performances(
        models, X_test, y_test, cluster_names
    )
    dl_results['model_performances'] = model_performances
    dl_results['best_model_name'] = best_model_name

    # 5. 최고 모델 상세 분석
    print(f"\n{'='*20} 5단계: 최고 모델 상세 분석 {'='*20}")
    best_model = models[best_model_name]

    # 혼동 행렬
    best_performance = model_performances[best_model_name]
    cm, cm_normalized = create_confusion_matrix_plot(
        y_test, best_performance['y_pred'], cluster_names
    )

    # 예측 신뢰도 분석
    confidence_scores, bin_accuracies = create_prediction_confidence_analysis(
        best_performance['y_pred_proba'], y_test, cluster_names
    )

    # 특성 중요도 분석
    feature_importance = analyze_feature_importance_with_permutation(
        best_model, X_test, y_test, classification_data['feature_names']
    )

    dl_results['detailed_analysis'] = {
        'confusion_matrix': cm_normalized,
        'confidence_analysis': confidence_scores,
        'feature_importance': feature_importance
    }

    # 6. 앙상블 모델
    print(f"\n{'='*20} 6단계: 앙상블 모델 {'='*20}")
    ensemble_pred, ensemble_proba, ensemble_accuracy = create_ensemble_model(
        models, X_test, y_test
    )

    dl_results['ensemble'] = {
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'accuracy': ensemble_accuracy
    }

    # 실행 시간 계산
    execution_time = time.time() - start_time

    # 최종 요약
    print(f"\n{'='*20} 딥러닝 파이프라인 완료 요약 {'='*20}")
    print(f"⏰ 실행 시간: {execution_time:.1f}초")
    print(f"🎯 훈련된 모델 수: {len(models)}개")
    print(f"🏆 최고 성능 모델: {best_model_name}")
    print(f"📊 최고 모델 성능:")
    best_perf = model_performances[best_model_name]
    print(f"  정확도: {best_perf['accuracy']:.4f}")
    print(f"  F1 (Macro): {best_perf['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"🤝 앙상블 모델 성능: {ensemble_accuracy:.4f}")

    # 성능 향상도 계산
    improvement = ensemble_accuracy - best_perf['accuracy']
    print(f"🚀 앙상블 향상도: {improvement:+.4f} ({improvement*100:+.2f}%)")

    print(f"✅ Phase 6 완료! Phase 7 (모델 해석)로 진행 가능")

    return dl_results
```
### 7. 실행 함수
``` python
# 7. 실행 함수

def run_phase6_pipeline(modeling_data, clustering_results):
    """Phase 6 전체 파이프라인 실행"""
    print("🎮 PUBG Phase 6: 지도 학습 (딥러닝 분류) 시작!")
    print("="*60)

    # 메모리 확인
    check_memory_usage()

    # GPU 사용 확인
    print(f"🔧 GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # 종합 딥러닝 파이프라인 실행
    dl_results = comprehensive_deep_learning_pipeline(modeling_data, clustering_results)

    return dl_results
```
### 8. 모델 저장 및 배포 준비
``` python
# 8. 모델 저장 및 배포 준비

def save_best_model(dl_results, save_path='pubg_best_model'):
    """최고 성능 모델 저장"""
    print(f"\n💾 최고 성능 모델 저장")
    print("-" * 40)

    best_model_name = dl_results['best_model_name']
    best_model = dl_results['models'][best_model_name]

    # 모델 저장
    best_model.save(f'{save_path}.h5')
    print(f"✅ 모델 저장 완료: {save_path}.h5")

    # 메타데이터 저장
    metadata = {
        'model_name': best_model_name,
        'accuracy': dl_results['model_performances'][best_model_name]['accuracy'],
        'feature_names': dl_results['classification_data']['feature_names'],
        'cluster_names': dl_results['classification_data']['cluster_names'],
        'n_classes': dl_results['classification_data']['n_classes']
    }

    import json
    with open(f'{save_path}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ 메타데이터 저장 완료: {save_path}_metadata.json")

    return save_path

def create_prediction_function(model_path, metadata_path):
    """새로운 플레이어 예측 함수 생성"""
    print(f"\n🎯 예측 함수 생성")
    print("-" * 40)

    # 모델 및 메타데이터 로드
    model = keras.models.load_model(model_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    def predict_player_type(player_features):
        """
        새로운 플레이어의 유형을 예측하는 함수

        Args:
            player_features: 플레이어 특성 벡터 (numpy array 또는 list)

        Returns:
            dict: 예측 결과 (클러스터 이름, 확률 등)
        """
        # 예측 수행
        prediction_proba = model.predict(np.array([player_features]), verbose=0)[0]
        predicted_cluster = np.argmax(prediction_proba)
        confidence = np.max(prediction_proba)

        # 결과 반환
        cluster_names = metadata['cluster_names']
        predicted_name = cluster_names.get(str(predicted_cluster), f'Cluster_{predicted_cluster}')

        return {
            'predicted_cluster': int(predicted_cluster),
            'predicted_type': predicted_name,
            'confidence': float(confidence),
            'all_probabilities': {
                cluster_names.get(str(i), f'Cluster_{i}'): float(prob)
                for i, prob in enumerate(prediction_proba)
            }
        }

    print(f"✅ 예측 함수 생성 완료")
    print(f"📊 모델 정보:")
    print(f"  특성 수: {len(metadata['feature_names'])}")
    print(f"  클러스터 수: {metadata['n_classes']}")
    print(f"  모델 정확도: {metadata['accuracy']:.4f}")

    return predict_player_type
```
### 9. 실행 가이드
``` python
# 9. 실행 가이드

def display_phase6_guide():
    """Phase 6 실행 가이드 표시"""
    print("\n📋 Phase 6 실행 방법:")
    print("="*50)
    print("# 1. Phase 4, 5에서 준비된 데이터 사용")
    print("dl_results = run_phase6_pipeline(modeling_data, clustering_results)")
    print()
    print("# 2. 결과 확인")
    print("print('최고 모델:', dl_results['best_model_name'])")
    print("print('최고 성능:', dl_results['model_performances'][dl_results['best_model_name']]['accuracy'])")
    print("print('앙상블 성능:', dl_results['ensemble']['accuracy'])")
    print()
    print("# 3. 모델 저장")
    print("model_path = save_best_model(dl_results)")
    print()
    print("# 4. 예측 함수 생성")
    print("predict_fn = create_prediction_function(f'{model_path}.h5', f'{model_path}_metadata.json')")

# 가이드 표시
display_phase6_guide()

print("\n✅ Phase 6 준비 완료!")
print("🧠 다음 단계: Phase 7 - 모델 해석 및 고도화")
```
### 10. 추가 유틸리티 함수들
``` python
# 10. 추가 유틸리티 함수들

def create_model_comparison_report(dl_results):
    """모델 비교 리포트 생성"""
    print(f"\n📊 모델 비교 리포트")
    print("="*50)

    performances = dl_results['model_performances']

    # 성능 테이블 생성
    comparison_data = []
    for model_name, perf in performances.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{perf['accuracy']:.4f}",
            'F1_Macro': f"{perf['classification_report']['macro avg']['f1-score']:.4f}",
            'F1_Weighted': f"{perf['classification_report']['weighted avg']['f1-score']:.4f}",
            'Precision_Macro': f"{perf['classification_report']['macro avg']['precision']:.4f}",
            'Recall_Macro': f"{perf['classification_report']['macro avg']['recall']:.4f}"
        })

    # 앙상블 모델 추가
    if 'ensemble' in dl_results:
        comparison_data.append({
            'Model': 'Ensemble',
            'Accuracy': f"{dl_results['ensemble']['accuracy']:.4f}",
            'F1_Macro': 'N/A',
            'F1_Weighted': 'N/A',
            'Precision_Macro': 'N/A',
            'Recall_Macro': 'N/A'
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    return comparison_df

def analyze_misclassified_samples(dl_results, top_n=10):
    """잘못 분류된 샘플 분석"""
    print(f"\n🔍 잘못 분류된 샘플 분석 (상위 {top_n}개)")
    print("-" * 40)

    best_model_name = dl_results['best_model_name']
    best_performance = dl_results['model_performances'][best_model_name]

    y_test = dl_results['classification_data']['y_test']
    y_pred = best_performance['y_pred']
    y_pred_proba = best_performance['y_pred_proba']
    cluster_names = dl_results['classification_data']['cluster_names']

    # 잘못 분류된 샘플 찾기
    misclassified_mask = y_test != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]

    # 신뢰도가 높은 잘못된 예측들 (가장 문제가 되는 케이스)
    misclassified_confidence = np.max(y_pred_proba[misclassified_mask], axis=1)

    # 상위 N개 추출
    top_misclassified_idx = np.argsort(misclassified_confidence)[-top_n:]

    print(f"📊 총 잘못 분류된 샘플: {len(misclassified_indices):,}개")
    print(f"🎯 상위 {top_n}개 높은 신뢰도 오분류:")

    for i, idx in enumerate(top_misclassified_idx):
        original_idx = misclassified_indices[idx]
        true_label = y_test[original_idx]
        pred_label = y_pred[original_idx]
        confidence = misclassified_confidence[idx]

        true_name = cluster_names.get(true_label, f'Cluster_{true_label}')
        pred_name = cluster_names.get(pred_label, f'Cluster_{pred_label}')

        print(f"  {i+1:2d}. 실제: {true_name} → 예측: {pred_name} (신뢰도: {confidence:.3f})")

    return misclassified_indices

print("\n🎯 Phase 6 준비 완료!")
print("📊 딥러닝 분류 모델의 모든 기능이 포함된 완전한 파이프라인입니다.")
```
### 실행
``` python
# Phase 6 실행 전 필요한 변수들 확인 및 설정

print("🔍 Phase 6 실행 전 변수 상태 확인")
print("="*50)

# 1. 필요한 변수들 존재 여부 확인
required_vars = ['modeling_data', 'clustering_results', 'final_results']

for var_name in required_vars:
    if var_name in globals():
        print(f"✅ {var_name}: 존재함")
    else:
        print(f"❌ {var_name}: 존재하지 않음")

print("\n" + "="*50)

# 2. clustering_results가 없는 경우 final_results에서 가져오기
if 'clustering_results' not in globals():
    if 'final_results' in globals():
        print("🔄 final_results에서 clustering_results 추출 중...")
        clustering_results = final_results
        print("✅ clustering_results 설정 완료")
    else:
        print("❌ clustering_results와 final_results 모두 없음")
        print("🔄 Phase 5를 다시 실행해야 합니다.")

        # Phase 5 간단 재실행 (clustering_results 생성용)
        print("\n🚀 Phase 5 간단 재실행 시작...")

        # modeling_data가 있는지 확인
        if 'modeling_data' not in globals():
            print("❌ modeling_data도 없습니다. Phase 4부터 다시 실행해야 합니다.")
            print("🔄 Phase 4 간단 재실행...")

            # Phase 4 최소 실행 (modeling_data 생성용)
            if 'df_cleaned' in globals():
                # Phase 4 최소 실행
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                print("📊 modeling_data 간단 생성 중...")

                # 특성과 타겟 분리
                feature_cols = [col for col in df_cleaned.columns if col not in ['Id', 'groupId', 'matchId', 'winPlacePerc']]
                X = df_cleaned[feature_cols].fillna(0)
                y = df_cleaned['winPlacePerc'].fillna(df_cleaned['winPlacePerc'].median())

                # 훈련/테스트 분할
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # 스케일링
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

                modeling_data = {
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,
                    'y_train': y_train,
                    'y_test': y_test,
                    'feature_names': feature_cols
                }

                print(f"✅ modeling_data 생성 완료: {X_train_scaled.shape}")
            else:
                print("❌ df_cleaned도 없습니다. Phase 1-3을 먼저 실행하세요.")
                raise NameError("필요한 데이터가 없습니다. Phase 1-3을 먼저 실행하세요.")

        # Phase 5 간단 실행 (클러스터링만)
        print("\n🎯 클러스터링 간단 실행...")

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # K-Means 클러스터링 (간단 버전)
        X_train = modeling_data['X_train']

        # 최적 클러스터 수 간단 결정 (5개로 고정)
        n_clusters = 5

        print(f"🔄 K-Means 클러스터링 (K={n_clusters}) 실행 중...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_train)

        # 클러스터 이름 생성
        cluster_names = {
            0: "Aggressive Fighter",
            1: "Cautious Survivor",
            2: "Mobile Explorer",
            3: "Team Supporter",
            4: "Balanced Player"
        }

        # 클러스터 프로파일 생성
        cluster_profiles = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_percentage = (cluster_size / len(cluster_labels)) * 100

            if cluster_size > 0:
                cluster_data = X_train[cluster_mask]
                overall_mean = X_train.mean()
                cluster_mean = cluster_data.mean()
                feature_ratios = cluster_mean / (overall_mean + 1e-8)

                cluster_profiles[cluster_id] = {
                    'size': cluster_size,
                    'percentage': cluster_percentage,
                    'top_features': feature_ratios.nlargest(5),
                    'bottom_features': feature_ratios.nsmallest(5),
                    'mean_values': cluster_mean
                }

        # clustering_results 구성
        clustering_results = {
            'kmeans_model': kmeans,
            'cluster_labels': cluster_labels,
            'cluster_names': cluster_names,
            'cluster_profiles': cluster_profiles,
            'optimal_clusters': {'final_decision': n_clusters},
            'clustering_metrics': {
                'silhouette_score': silhouette_score(X_train, cluster_labels),
                'inertia': kmeans.inertia_
            }
        }

        print(f"✅ clustering_results 생성 완료")
        print(f"📊 클러스터 수: {n_clusters}")
        print(f"🎯 Silhouette Score: {clustering_results['clustering_metrics']['silhouette_score']:.4f}")

# 3. 변수 재확인
print(f"\n🔍 변수 재확인:")
for var_name in required_vars:
    if var_name in globals():
        if var_name == 'modeling_data':
            print(f"✅ {var_name}: {type(globals()[var_name])}, 키: {list(globals()[var_name].keys())}")
        elif var_name == 'clustering_results':
            print(f"✅ {var_name}: {type(globals()[var_name])}, 키: {list(globals()[var_name].keys())}")
        else:
            print(f"✅ {var_name}: 존재함")
    else:
        print(f"❌ {var_name}: 여전히 없음")

print(f"\n🎮 Phase 6 실행 준비 완료!")
print("="*50)

# 4. Phase 6 실행
try:
    print("🚀 Phase 6 딥러닝 파이프라인 시작!")

    # Phase 6 실행
    dl_results = run_phase6_pipeline(modeling_data, clustering_results)

    # 결과 확인
    print('\n📊 Phase 6 실행 결과:')
    print('최고 모델:', dl_results['best_model_name'])
    print('최고 성능:', f"{dl_results['model_performances'][dl_results['best_model_name']]['accuracy']:.4f}")
    print('앙상블 성능:', f"{dl_results['ensemble']['accuracy']:.4f}")

    # 모델 저장
    print("\n💾 모델 저장 중...")
    model_path = save_best_model(dl_results)

    # 예측 함수 생성
    print("🔮 예측 함수 생성 중...")
    predict_fn = create_prediction_function(f'{model_path}.h5', f'{model_path}_metadata.json')

    print("\n🎉 Phase 6 완료!")
    print("✅ 다음 단계: Phase 7 - 모델 해석 및 고도화")

except Exception as e:
    print(f"❌ Phase 6 실행 중 오류 발생: {e}")
    print(f"📋 오류 상세:")
    import traceback
    traceback.print_exc()

    print(f"\n💡 문제 해결 방안:")
    print(f"1. Phase 1-5를 순서대로 다시 실행")
    print(f"2. 메모리 부족시 샘플 크기 줄이기")
    print(f"3. GPU 런타임 사용 권장")

print(f"\n📝 Phase 6 사용 가능한 함수들:")
print(f"• dl_results: 딥러닝 결과")
print(f"• predict_fn: 새 플레이어 예측 함수")
print(f"• model_path: 저장된 모델 경로")
```

#### 실행 결과

``` bash
🔍 Phase 6 실행 전 변수 상태 확인
==================================================
✅ modeling_data: 존재함
❌ clustering_results: 존재하지 않음
✅ final_results: 존재함

==================================================
🔄 final_results에서 clustering_results 추출 중...
✅ clustering_results 설정 완료

🔍 변수 재확인:
✅ modeling_data: <class 'dict'>, 키: ['X_train', 'X_test', 'y_train', 'y_test', 'feature_names']
✅ clustering_results: <class 'dict'>, 키: ['optimal_clusters', 'kmeans_model', 'cluster_labels', 'clustering_metrics', 'cluster_profiles', 'cluster_stats', 'visualization_data', 'cluster_names']
✅ final_results: 존재함
```

``` bash
🎮 Phase 6 실행 준비 완료!
==================================================
🚀 Phase 6 딥러닝 파이프라인 시작!
🎮 PUBG Phase 6: 지도 학습 (딥러닝 분류) 시작!
============================================================
💾 메모리 사용량: 30.3% (3.5GB / 12.7GB)
🔧 GPU 사용 가능: False
🚀 종합 딥러닝 분류 파이프라인 시작
============================================================

==================== 1단계: 분류 데이터 준비 ====================
🎯 분류 모델용 데이터 준비
----------------------------------------
📊 분류 데이터 정보:
  훈련 특성: (80000, 30)
  테스트 특성: (20000, 30)
  클러스터 수: 8개
  클러스터 분포 (훈련):
    Survivor: 14,527개 (18.2%)
    Survivor: 24,981개 (31.2%)
    Explorer: 10,756개 (13.4%)
    Explorer: 15,898개 (19.9%)
    Explorer: 4,312개 (5.4%)
    Explorer: 4,046개 (5.1%)
    Explorer: 5,391개 (6.7%)
    Aggressive: 89개 (0.1%)
```

``` bash
==================== 2단계: 다중 모델 훈련 ====================

🏆 다중 모델 비교 분석
==================================================
🔀 검증 데이터 분할 (20%)
----------------------------------------
📊 분할 결과:
  훈련 세트: 64,000개
  검증 세트: 16,000개

==================== 기본 신경망 모델 훈련 ====================
🧠 기본 신경망 모델 구축
  입력 차원: 30
  출력 클래스: 8
  은닉층: [128, 64, 32]
----------------------------------------
🎯 모델 훈련 시작
  에포크: 50
  배치 크기: 256
  학습률: 0.001
----------------------------------------
🏗️ 모델 아키텍처:
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │         3,968 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 128)            │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 64)             │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 32)             │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 32)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 8)              │           264 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 15,464 (60.41 KB)
 Trainable params: 15,016 (58.66 KB)
 Non-trainable params: 448 (1.75 KB)
```

``` bash
🔄 모델 훈련 진행 중...
Epoch 1/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 8ms/step - accuracy: 0.6824 - loss: 0.9892 - val_accuracy: 0.9664 - val_loss: 0.1198 - learning_rate: 0.0010
Epoch 2/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9264 - loss: 0.2040 - val_accuracy: 0.9803 - val_loss: 0.0682 - learning_rate: 0.0010
Epoch 3/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9452 - loss: 0.1465 - val_accuracy: 0.9826 - val_loss: 0.0550 - learning_rate: 0.0010
Epoch 4/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9541 - loss: 0.1203 - val_accuracy: 0.9839 - val_loss: 0.0479 - learning_rate: 0.0010
Epoch 5/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9595 - loss: 0.1054 - val_accuracy: 0.9861 - val_loss: 0.0426 - learning_rate: 0.0010
Epoch 6/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9636 - loss: 0.0978 - val_accuracy: 0.9829 - val_loss: 0.0427 - learning_rate: 0.0010
Epoch 7/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.9650 - loss: 0.0912 - val_accuracy: 0.9858 - val_loss: 0.0388 - learning_rate: 0.0010
Epoch 8/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - accuracy: 0.9692 - loss: 0.0805 - val_accuracy: 0.9892 - val_loss: 0.0343 - learning_rate: 0.0010
Epoch 9/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9690 - loss: 0.0792 - val_accuracy: 0.9892 - val_loss: 0.0340 - learning_rate: 0.0010
Epoch 10/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9705 - loss: 0.0761 - val_accuracy: 0.9845 - val_loss: 0.0379 - learning_rate: 0.0010
Epoch 11/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9723 - loss: 0.0727 - val_accuracy: 0.9881 - val_loss: 0.0334 - learning_rate: 0.0010
Epoch 12/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9738 - loss: 0.0665 - val_accuracy: 0.9888 - val_loss: 0.0321 - learning_rate: 0.0010
Epoch 13/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9749 - loss: 0.0658 - val_accuracy: 0.9904 - val_loss: 0.0294 - learning_rate: 0.0010
Epoch 14/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9756 - loss: 0.0632 - val_accuracy: 0.9892 - val_loss: 0.0310 - learning_rate: 0.0010
Epoch 15/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.9751 - loss: 0.0639 - val_accuracy: 0.9851 - val_loss: 0.0344 - learning_rate: 0.0010
Epoch 16/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - accuracy: 0.9766 - loss: 0.0600 - val_accuracy: 0.9875 - val_loss: 0.0326 - learning_rate: 0.0010
Epoch 17/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9765 - loss: 0.0611 - val_accuracy: 0.9843 - val_loss: 0.0356 - learning_rate: 0.0010
Epoch 18/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9775 - loss: 0.0566 - val_accuracy: 0.9889 - val_loss: 0.0307 - learning_rate: 0.0010
Epoch 19/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9781 - loss: 0.0571 - val_accuracy: 0.9869 - val_loss: 0.0315 - learning_rate: 0.0010
Epoch 20/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9790 - loss: 0.0563 - val_accuracy: 0.9895 - val_loss: 0.0274 - learning_rate: 0.0010
Epoch 21/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9792 - loss: 0.0530 - val_accuracy: 0.9873 - val_loss: 0.0317 - learning_rate: 0.0010
Epoch 22/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.9798 - loss: 0.0529 - val_accuracy: 0.9891 - val_loss: 0.0283 - learning_rate: 0.0010
Epoch 23/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - accuracy: 0.9793 - loss: 0.0526 - val_accuracy: 0.9906 - val_loss: 0.0271 - learning_rate: 0.0010
Epoch 24/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9793 - loss: 0.0545 - val_accuracy: 0.9912 - val_loss: 0.0247 - learning_rate: 0.0010
Epoch 25/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9804 - loss: 0.0507 - val_accuracy: 0.9875 - val_loss: 0.0308 - learning_rate: 0.0010
Epoch 26/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9796 - loss: 0.0528 - val_accuracy: 0.9903 - val_loss: 0.0255 - learning_rate: 0.0010
Epoch 27/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9806 - loss: 0.0517 - val_accuracy: 0.9911 - val_loss: 0.0246 - learning_rate: 0.0010
Epoch 28/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9823 - loss: 0.0458 - val_accuracy: 0.9896 - val_loss: 0.0264 - learning_rate: 0.0010
Epoch 29/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9812 - loss: 0.0492 - val_accuracy: 0.9887 - val_loss: 0.0267 - learning_rate: 0.0010
Epoch 30/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9813 - loss: 0.0489 - val_accuracy: 0.9916 - val_loss: 0.0241 - learning_rate: 0.0010
Epoch 31/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9836 - loss: 0.0442 - val_accuracy: 0.9860 - val_loss: 0.0317 - learning_rate: 0.0010
Epoch 32/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9825 - loss: 0.0461 - val_accuracy: 0.9900 - val_loss: 0.0277 - learning_rate: 0.0010
Epoch 33/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9822 - loss: 0.0453 - val_accuracy: 0.9894 - val_loss: 0.0256 - learning_rate: 0.0010
Epoch 34/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9820 - loss: 0.0457 - val_accuracy: 0.9896 - val_loss: 0.0253 - learning_rate: 0.0010
Epoch 35/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9820 - loss: 0.0469 - val_accuracy: 0.9905 - val_loss: 0.0239 - learning_rate: 0.0010
Epoch 36/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9829 - loss: 0.0442 - val_accuracy: 0.9889 - val_loss: 0.0266 - learning_rate: 0.0010
Epoch 37/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.9842 - loss: 0.0422 - val_accuracy: 0.9918 - val_loss: 0.0214 - learning_rate: 0.0010
Epoch 38/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.9828 - loss: 0.0444 - val_accuracy: 0.9899 - val_loss: 0.0255 - learning_rate: 0.0010
Epoch 39/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.9844 - loss: 0.0416 - val_accuracy: 0.9908 - val_loss: 0.0227 - learning_rate: 0.0010
Epoch 40/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9842 - loss: 0.0409 - val_accuracy: 0.9932 - val_loss: 0.0196 - learning_rate: 0.0010
Epoch 41/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9856 - loss: 0.0376 - val_accuracy: 0.9898 - val_loss: 0.0249 - learning_rate: 0.0010
Epoch 42/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9845 - loss: 0.0404 - val_accuracy: 0.9876 - val_loss: 0.0288 - learning_rate: 0.0010
Epoch 43/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9848 - loss: 0.0408 - val_accuracy: 0.9911 - val_loss: 0.0229 - learning_rate: 0.0010
Epoch 44/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9849 - loss: 0.0408 - val_accuracy: 0.9908 - val_loss: 0.0239 - learning_rate: 0.0010
Epoch 45/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.9849 - loss: 0.0417 - val_accuracy: 0.9889 - val_loss: 0.0258 - learning_rate: 0.0010
Epoch 46/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.9836 - loss: 0.0435 - val_accuracy: 0.9908 - val_loss: 0.0221 - learning_rate: 0.0010
Epoch 47/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step - accuracy: 0.9852 - loss: 0.0384 - val_accuracy: 0.9893 - val_loss: 0.0245 - learning_rate: 0.0010
Epoch 48/50
246/250 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9865 - loss: 0.0355
Epoch 48: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9865 - loss: 0.0355 - val_accuracy: 0.9918 - val_loss: 0.0215 - learning_rate: 0.0010
Epoch 49/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9870 - loss: 0.0346 - val_accuracy: 0.9939 - val_loss: 0.0174 - learning_rate: 5.0000e-04
Epoch 50/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.9868 - loss: 0.0338 - val_accuracy: 0.9927 - val_loss: 0.0181 - learning_rate: 5.0000e-04
Restoring model weights from the end of the best epoch: 49.
⏰ 훈련 완료 시간: 90.6초
```

``` bash
==================== 고급 신경망 모델 훈련 ====================
🚀 고급 신경망 모델 구축
  입력 차원: 30
  출력 클래스: 8
----------------------------------------
🎯 모델 훈련 시작
  에포크: 50
  배치 크기: 256
  학습률: 0.001
----------------------------------------
🏗️ 모델 아키텍처:
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_1       │ (None, 30)        │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_4 (Dense)     │ (None, 256)       │      7,936 │ input_layer_1[0]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_6 (Dense)     │ (None, 128)       │      3,968 │ input_layer_1[0]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 256)       │      1,024 │ dense_4[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 128)       │        512 │ dense_6[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_3 (Dropout) │ (None, 256)       │          0 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_5 (Dropout) │ (None, 128)       │          0 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_5 (Dense)     │ (None, 128)       │     32,896 │ dropout_3[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_7 (Dense)     │ (None, 64)        │      8,256 │ dropout_5[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 128)       │        512 │ dense_5[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 64)        │        256 │ dense_7[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_4 (Dropout) │ (None, 128)       │          0 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_6 (Dropout) │ (None, 64)        │          0 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (None, 192)       │          0 │ dropout_4[0][0],  │
│ (Concatenate)       │                   │            │ dropout_6[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_8 (Dense)     │ (None, 64)        │     12,352 │ concatenate[0][0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 64)        │        256 │ dense_8[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_7 (Dropout) │ (None, 64)        │          0 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_9 (Dense)     │ (None, 32)        │      2,080 │ dropout_7[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_8 (Dropout) │ (None, 32)        │          0 │ dense_9[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_10 (Dense)    │ (None, 8)         │        264 │ dropout_8[0][0]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 70,312 (274.66 KB)
 Trainable params: 69,032 (269.66 KB)
 Non-trainable params: 1,280 (5.00 KB)
```

``` bash
🔄 모델 훈련 진행 중...
Epoch 1/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 9s 12ms/step - accuracy: 0.7798 - loss: 0.6758 - val_accuracy: 0.9743 - val_loss: 0.0793 - learning_rate: 0.0010
Epoch 2/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9390 - loss: 0.1599 - val_accuracy: 0.9817 - val_loss: 0.0541 - learning_rate: 0.0010
Epoch 3/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9541 - loss: 0.1170 - val_accuracy: 0.9650 - val_loss: 0.0839 - learning_rate: 0.0010
Epoch 4/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - accuracy: 0.9567 - loss: 0.1104 - val_accuracy: 0.9818 - val_loss: 0.0464 - learning_rate: 0.0010
Epoch 5/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9618 - loss: 0.0981 - val_accuracy: 0.9841 - val_loss: 0.0411 - learning_rate: 0.0010
Epoch 6/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9662 - loss: 0.0891 - val_accuracy: 0.9879 - val_loss: 0.0378 - learning_rate: 0.0010
Epoch 7/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9666 - loss: 0.0851 - val_accuracy: 0.9797 - val_loss: 0.0461 - learning_rate: 0.0010
Epoch 8/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9682 - loss: 0.0797 - val_accuracy: 0.9833 - val_loss: 0.0420 - learning_rate: 0.0010
Epoch 9/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.9703 - loss: 0.0737 - val_accuracy: 0.9862 - val_loss: 0.0370 - learning_rate: 0.0010
Epoch 10/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 15ms/step - accuracy: 0.9720 - loss: 0.0694 - val_accuracy: 0.9846 - val_loss: 0.0380 - learning_rate: 0.0010
Epoch 11/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - accuracy: 0.9705 - loss: 0.0724 - val_accuracy: 0.9843 - val_loss: 0.0377 - learning_rate: 0.0010
Epoch 12/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9732 - loss: 0.0674 - val_accuracy: 0.9887 - val_loss: 0.0335 - learning_rate: 0.0010
Epoch 13/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9751 - loss: 0.0636 - val_accuracy: 0.9854 - val_loss: 0.0334 - learning_rate: 0.0010
Epoch 14/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 15ms/step - accuracy: 0.9763 - loss: 0.0615 - val_accuracy: 0.9864 - val_loss: 0.0340 - learning_rate: 0.0010
Epoch 15/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9750 - loss: 0.0618 - val_accuracy: 0.9873 - val_loss: 0.0337 - learning_rate: 0.0010
Epoch 16/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9766 - loss: 0.0564 - val_accuracy: 0.9898 - val_loss: 0.0273 - learning_rate: 0.0010
Epoch 17/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9771 - loss: 0.0563 - val_accuracy: 0.9891 - val_loss: 0.0287 - learning_rate: 0.0010
Epoch 18/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - accuracy: 0.9779 - loss: 0.0557 - val_accuracy: 0.9901 - val_loss: 0.0283 - learning_rate: 0.0010
Epoch 19/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - accuracy: 0.9799 - loss: 0.0521 - val_accuracy: 0.9878 - val_loss: 0.0335 - learning_rate: 0.0010
Epoch 20/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.9781 - loss: 0.0553 - val_accuracy: 0.9891 - val_loss: 0.0300 - learning_rate: 0.0010
Epoch 21/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9799 - loss: 0.0509 - val_accuracy: 0.9829 - val_loss: 0.0382 - learning_rate: 0.0010
Epoch 22/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9806 - loss: 0.0489 - val_accuracy: 0.9902 - val_loss: 0.0281 - learning_rate: 0.0010
Epoch 23/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 16ms/step - accuracy: 0.9803 - loss: 0.0486 - val_accuracy: 0.9912 - val_loss: 0.0262 - learning_rate: 0.0010
Epoch 24/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9811 - loss: 0.0493 - val_accuracy: 0.9897 - val_loss: 0.0292 - learning_rate: 0.0010
Epoch 25/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.9817 - loss: 0.0456 - val_accuracy: 0.9894 - val_loss: 0.0276 - learning_rate: 0.0010
Epoch 26/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 16ms/step - accuracy: 0.9829 - loss: 0.0453 - val_accuracy: 0.9899 - val_loss: 0.0260 - learning_rate: 0.0010
Epoch 27/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9818 - loss: 0.0456 - val_accuracy: 0.9900 - val_loss: 0.0263 - learning_rate: 0.0010
Epoch 28/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9830 - loss: 0.0442 - val_accuracy: 0.9887 - val_loss: 0.0276 - learning_rate: 0.0010
Epoch 29/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.9808 - loss: 0.0451 - val_accuracy: 0.9875 - val_loss: 0.0309 - learning_rate: 0.0010
Epoch 30/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.9826 - loss: 0.0457 - val_accuracy: 0.9895 - val_loss: 0.0264 - learning_rate: 0.0010
Epoch 31/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - accuracy: 0.9817 - loss: 0.0450 - val_accuracy: 0.9879 - val_loss: 0.0284 - learning_rate: 0.0010
Epoch 32/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 16ms/step - accuracy: 0.9833 - loss: 0.0398 - val_accuracy: 0.9894 - val_loss: 0.0258 - learning_rate: 0.0010
Epoch 33/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9834 - loss: 0.0416 - val_accuracy: 0.9890 - val_loss: 0.0262 - learning_rate: 0.0010
Epoch 34/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - accuracy: 0.9841 - loss: 0.0394 - val_accuracy: 0.9927 - val_loss: 0.0237 - learning_rate: 0.0010
Epoch 35/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 15ms/step - accuracy: 0.9847 - loss: 0.0384 - val_accuracy: 0.9907 - val_loss: 0.0237 - learning_rate: 0.0010
Epoch 36/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9839 - loss: 0.0377 - val_accuracy: 0.9910 - val_loss: 0.0232 - learning_rate: 0.0010
Epoch 37/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9847 - loss: 0.0386 - val_accuracy: 0.9888 - val_loss: 0.0270 - learning_rate: 0.0010
Epoch 38/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.9847 - loss: 0.0398 - val_accuracy: 0.9919 - val_loss: 0.0227 - learning_rate: 0.0010
Epoch 39/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9849 - loss: 0.0379 - val_accuracy: 0.9887 - val_loss: 0.0300 - learning_rate: 0.0010
Epoch 40/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 12ms/step - accuracy: 0.9834 - loss: 0.0391 - val_accuracy: 0.9912 - val_loss: 0.0248 - learning_rate: 0.0010
Epoch 41/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - accuracy: 0.9858 - loss: 0.0353 - val_accuracy: 0.9919 - val_loss: 0.0227 - learning_rate: 0.0010
Epoch 42/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - accuracy: 0.9848 - loss: 0.0369 - val_accuracy: 0.9909 - val_loss: 0.0237 - learning_rate: 0.0010
Epoch 43/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 6s 16ms/step - accuracy: 0.9872 - loss: 0.0337 - val_accuracy: 0.9906 - val_loss: 0.0234 - learning_rate: 0.0010
Epoch 44/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.9860 - loss: 0.0353 - val_accuracy: 0.9914 - val_loss: 0.0219 - learning_rate: 0.0010
Epoch 45/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 12ms/step - accuracy: 0.9856 - loss: 0.0353 - val_accuracy: 0.9923 - val_loss: 0.0216 - learning_rate: 0.0010
Epoch 46/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - accuracy: 0.9854 - loss: 0.0368 - val_accuracy: 0.9905 - val_loss: 0.0234 - learning_rate: 0.0010
Epoch 47/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9856 - loss: 0.0367 - val_accuracy: 0.9905 - val_loss: 0.0241 - learning_rate: 0.0010
Epoch 48/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - accuracy: 0.9853 - loss: 0.0365 - val_accuracy: 0.9918 - val_loss: 0.0217 - learning_rate: 0.0010
Epoch 49/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - accuracy: 0.9851 - loss: 0.0355 - val_accuracy: 0.9922 - val_loss: 0.0218 - learning_rate: 0.0010
Epoch 50/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9869 - loss: 0.0333 - val_accuracy: 0.9911 - val_loss: 0.0208 - learning_rate: 0.0010
Restoring model weights from the end of the best epoch: 50.
⏰ 훈련 완료 시간: 221.8초
```

``` bash
==================== Residual 신경망 모델 훈련 ====================
🔗 Residual 신경망 모델 구축
----------------------------------------
🎯 모델 훈련 시작
  에포크: 50
  배치 크기: 256
  학습률: 0.001
----------------------------------------
🏗️ 모델 아키텍처:
Model: "functional_2"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_2       │ (None, 30)        │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_11 (Dense)    │ (None, 128)       │      3,968 │ input_layer_2[0]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 128)       │        512 │ dense_11[0][0]    │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_12 (Dense)    │ (None, 128)       │     16,512 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 128)       │        512 │ dense_12[0][0]    │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_9 (Dropout) │ (None, 128)       │          0 │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_13 (Dense)    │ (None, 128)       │     16,512 │ dropout_9[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 128)       │        512 │ dense_13[0][0]    │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (Add)           │ (None, 128)       │          0 │ batch_normalizat… │
│                     │                   │            │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation          │ (None, 128)       │          0 │ add[0][0]         │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_15 (Dense)    │ (None, 64)        │      8,256 │ activation[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 64)        │        256 │ dense_15[0][0]    │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_10          │ (None, 64)        │          0 │ batch_normalizat… │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_16 (Dense)    │ (None, 64)        │      4,160 │ dropout_10[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 64)        │        256 │ dense_16[0][0]    │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_14 (Dense)    │ (None, 64)        │      8,256 │ activation[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (Add)         │ (None, 64)        │          0 │ batch_normalizat… │
│                     │                   │            │ dense_14[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ activation_1        │ (None, 64)        │          0 │ add_1[0][0]       │
│ (Activation)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_17 (Dense)    │ (None, 32)        │      2,080 │ activation_1[0][… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_11          │ (None, 32)        │          0 │ dense_17[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_18 (Dense)    │ (None, 8)         │        264 │ dropout_11[0][0]  │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 62,056 (242.41 KB)
 Trainable params: 61,032 (238.41 KB)
 Non-trainable params: 1,024 (4.00 KB)
 ```

``` bash
🔄 모델 훈련 진행 중...
Epoch 1/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 9s 17ms/step - accuracy: 0.7739 - loss: 0.6666 - val_accuracy: 0.9536 - val_loss: 0.1134 - learning_rate: 0.0010
Epoch 2/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9556 - loss: 0.1187 - val_accuracy: 0.9668 - val_loss: 0.0774 - learning_rate: 0.0010
Epoch 3/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.9583 - loss: 0.1035 - val_accuracy: 0.9775 - val_loss: 0.0552 - learning_rate: 0.0010
Epoch 4/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9677 - loss: 0.0822 - val_accuracy: 0.9787 - val_loss: 0.0505 - learning_rate: 0.0010
Epoch 5/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 14ms/step - accuracy: 0.9705 - loss: 0.0755 - val_accuracy: 0.9826 - val_loss: 0.0452 - learning_rate: 0.0010
Epoch 6/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - accuracy: 0.9723 - loss: 0.0701 - val_accuracy: 0.9778 - val_loss: 0.0503 - learning_rate: 0.0010
Epoch 7/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.9740 - loss: 0.0658 - val_accuracy: 0.9872 - val_loss: 0.0388 - learning_rate: 0.0010
Epoch 8/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9745 - loss: 0.0624 - val_accuracy: 0.9871 - val_loss: 0.0379 - learning_rate: 0.0010
Epoch 9/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.9755 - loss: 0.0620 - val_accuracy: 0.9834 - val_loss: 0.0403 - learning_rate: 0.0010
Epoch 10/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9757 - loss: 0.0605 - val_accuracy: 0.9834 - val_loss: 0.0394 - learning_rate: 0.0010
Epoch 11/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9781 - loss: 0.0551 - val_accuracy: 0.9819 - val_loss: 0.0434 - learning_rate: 0.0010
Epoch 12/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9774 - loss: 0.0549 - val_accuracy: 0.9772 - val_loss: 0.0572 - learning_rate: 0.0010
Epoch 13/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9801 - loss: 0.0506 - val_accuracy: 0.9867 - val_loss: 0.0361 - learning_rate: 0.0010
Epoch 14/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 15ms/step - accuracy: 0.9799 - loss: 0.0496 - val_accuracy: 0.9894 - val_loss: 0.0304 - learning_rate: 0.0010
Epoch 15/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9825 - loss: 0.0443 - val_accuracy: 0.9835 - val_loss: 0.0388 - learning_rate: 0.0010
Epoch 16/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9809 - loss: 0.0470 - val_accuracy: 0.9879 - val_loss: 0.0301 - learning_rate: 0.0010
Epoch 17/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9831 - loss: 0.0421 - val_accuracy: 0.9854 - val_loss: 0.0350 - learning_rate: 0.0010
Epoch 18/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9825 - loss: 0.0421 - val_accuracy: 0.9872 - val_loss: 0.0328 - learning_rate: 0.0010
Epoch 19/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 15ms/step - accuracy: 0.9842 - loss: 0.0397 - val_accuracy: 0.9853 - val_loss: 0.0347 - learning_rate: 0.0010
Epoch 20/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - accuracy: 0.9836 - loss: 0.0393 - val_accuracy: 0.9882 - val_loss: 0.0293 - learning_rate: 0.0010
Epoch 21/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9842 - loss: 0.0398 - val_accuracy: 0.9889 - val_loss: 0.0286 - learning_rate: 0.0010
Epoch 22/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9856 - loss: 0.0358 - val_accuracy: 0.9857 - val_loss: 0.0329 - learning_rate: 0.0010
Epoch 23/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 16ms/step - accuracy: 0.9844 - loss: 0.0384 - val_accuracy: 0.9831 - val_loss: 0.0379 - learning_rate: 0.0010
Epoch 24/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9855 - loss: 0.0357 - val_accuracy: 0.9898 - val_loss: 0.0258 - learning_rate: 0.0010
Epoch 25/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9855 - loss: 0.0352 - val_accuracy: 0.9896 - val_loss: 0.0257 - learning_rate: 0.0010
Epoch 26/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9864 - loss: 0.0334 - val_accuracy: 0.9851 - val_loss: 0.0345 - learning_rate: 0.0010
Epoch 27/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.9869 - loss: 0.0344 - val_accuracy: 0.9905 - val_loss: 0.0251 - learning_rate: 0.0010
Epoch 28/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - accuracy: 0.9865 - loss: 0.0327 - val_accuracy: 0.9887 - val_loss: 0.0261 - learning_rate: 0.0010
Epoch 29/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.9867 - loss: 0.0323 - val_accuracy: 0.9890 - val_loss: 0.0268 - learning_rate: 0.0010
Epoch 30/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9877 - loss: 0.0315 - val_accuracy: 0.9899 - val_loss: 0.0258 - learning_rate: 0.0010
Epoch 31/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.9881 - loss: 0.0297 - val_accuracy: 0.9891 - val_loss: 0.0256 - learning_rate: 0.0010
Epoch 32/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.9872 - loss: 0.0314 - val_accuracy: 0.9861 - val_loss: 0.0314 - learning_rate: 0.0010
Epoch 33/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.9869 - loss: 0.0309 - val_accuracy: 0.9899 - val_loss: 0.0274 - learning_rate: 0.0010
Epoch 34/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9899 - loss: 0.0264 - val_accuracy: 0.9904 - val_loss: 0.0252 - learning_rate: 0.0010
Epoch 35/50
246/250 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9886 - loss: 0.0287
Epoch 35: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9886 - loss: 0.0287 - val_accuracy: 0.9873 - val_loss: 0.0302 - learning_rate: 0.0010
Epoch 36/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9898 - loss: 0.0254 - val_accuracy: 0.9905 - val_loss: 0.0231 - learning_rate: 5.0000e-04
Epoch 37/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 13ms/step - accuracy: 0.9914 - loss: 0.0207 - val_accuracy: 0.9890 - val_loss: 0.0266 - learning_rate: 5.0000e-04
Epoch 38/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - accuracy: 0.9912 - loss: 0.0216 - val_accuracy: 0.9905 - val_loss: 0.0246 - learning_rate: 5.0000e-04
Epoch 39/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 9ms/step - accuracy: 0.9919 - loss: 0.0198 - val_accuracy: 0.9883 - val_loss: 0.0301 - learning_rate: 5.0000e-04
Epoch 40/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9920 - loss: 0.0195 - val_accuracy: 0.9875 - val_loss: 0.0336 - learning_rate: 5.0000e-04
Epoch 41/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9927 - loss: 0.0194 - val_accuracy: 0.9903 - val_loss: 0.0236 - learning_rate: 5.0000e-04
Epoch 42/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 5s 19ms/step - accuracy: 0.9913 - loss: 0.0212 - val_accuracy: 0.9912 - val_loss: 0.0231 - learning_rate: 5.0000e-04
Epoch 43/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.9925 - loss: 0.0189 - val_accuracy: 0.9914 - val_loss: 0.0215 - learning_rate: 5.0000e-04
Epoch 44/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9931 - loss: 0.0173 - val_accuracy: 0.9900 - val_loss: 0.0241 - learning_rate: 5.0000e-04
Epoch 45/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9933 - loss: 0.0185 - val_accuracy: 0.9908 - val_loss: 0.0238 - learning_rate: 5.0000e-04
Epoch 46/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 4s 14ms/step - accuracy: 0.9919 - loss: 0.0199 - val_accuracy: 0.9908 - val_loss: 0.0245 - learning_rate: 5.0000e-04
Epoch 47/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 3s 12ms/step - accuracy: 0.9931 - loss: 0.0170 - val_accuracy: 0.9901 - val_loss: 0.0242 - learning_rate: 5.0000e-04
Epoch 48/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.9930 - loss: 0.0169 - val_accuracy: 0.9896 - val_loss: 0.0271 - learning_rate: 5.0000e-04
Epoch 49/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9929 - loss: 0.0186 - val_accuracy: 0.9921 - val_loss: 0.0221 - learning_rate: 5.0000e-04
Epoch 50/50
250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9942 - loss: 0.0142 - val_accuracy: 0.9887 - val_loss: 0.0279 - learning_rate: 5.0000e-04
Restoring model weights from the end of the best epoch: 43.
⏰ 훈련 완료 시간: 154.9초
```

``` bash
==================== 3단계: 훈련 과정 시각화 ====================

📈 Basic 모델 훈련 과정:

📈 훈련 과정 시각화
----------------------------------------
```

[!ScreenShot](image/Basic모델_훈련_과정_훈련과정_시각화.png)

``` bash
📊 최종 훈련 성능:
  훈련 정확도: 0.9869
  검증 정확도: 0.9927
  훈련 손실: 0.0347
  검증 손실: 0.0181
✅ 적절한 일반화 성능 (차이: -0.006)

📈 Advanced 모델 훈련 과정:

📈 훈련 과정 시각화
----------------------------------------
```

[!ScreenShot](image/Advanced모델_훈련과정_시각화.png)

``` bash

📊 최종 훈련 성능:
  훈련 정확도: 0.9871
  검증 정확도: 0.9911
  훈련 손실: 0.0327
  검증 손실: 0.0208
✅ 적절한 일반화 성능 (차이: -0.004)

📈 Residual 모델 훈련 과정:

📈 훈련 과정 시각화
----------------------------------------
```

[!ScreenShot](image/Residual모델_훈련과정_시각화.png)

``` bash
📊 최종 훈련 성능:
  훈련 정확도: 0.9934
  검증 정확도: 0.9887
  훈련 손실: 0.0167
  검증 손실: 0.0279
✅ 적절한 일반화 성능 (차이: 0.005)

==================== 4단계: 모델 성능 평가 ====================

📊 모델 성능 비교
==================================================

🔍 Basic 모델 평가:

📊 모델 성능 평가
----------------------------------------
🎯 정확도: 0.9925 (99.25%)

📈 클래스별 성능:
  Survivor       : 정밀도=0.997, 재현율=0.999, F1=0.998
  Survivor       : 정밀도=0.997, 재현율=0.999, F1=0.998
  Explorer       : 정밀도=0.992, 재현율=0.967, F1=0.979
  Explorer       : 정밀도=0.992, 재현율=0.967, F1=0.979
  Explorer       : 정밀도=0.992, 재현율=0.967, F1=0.979
  Explorer       : 정밀도=0.992, 재현율=0.967, F1=0.979
  Explorer       : 정밀도=0.992, 재현율=0.967, F1=0.979
  Aggressive     : 정밀도=0.950, 재현율=1.000, F1=0.974

📊 전체 평균:
  Macro Avg    : F1=0.987
  Weighted Avg : F1=0.992
  정확도: 0.9925
  F1 (Macro): 0.9867

🔍 Advanced 모델 평가:

📊 모델 성능 평가
----------------------------------------
🎯 정확도: 0.9896 (98.96%)

📈 클래스별 성능:
  Survivor       : 정밀도=0.995, 재현율=0.999, F1=0.997
  Survivor       : 정밀도=0.995, 재현율=0.999, F1=0.997
  Explorer       : 정밀도=0.999, 재현율=0.939, F1=0.968
  Explorer       : 정밀도=0.999, 재현율=0.939, F1=0.968
  Explorer       : 정밀도=0.999, 재현율=0.939, F1=0.968
  Explorer       : 정밀도=0.999, 재현율=0.939, F1=0.968
  Explorer       : 정밀도=0.999, 재현율=0.939, F1=0.968
  Aggressive     : 정밀도=1.000, 재현율=1.000, F1=1.000

📊 전체 평균:
  Macro Avg    : F1=0.987
  Weighted Avg : F1=0.990
  정확도: 0.9896
  F1 (Macro): 0.9870

🔍 Residual 모델 평가:

📊 모델 성능 평가
----------------------------------------
🎯 정확도: 0.9899 (99.00%)

📈 클래스별 성능:
  Survivor       : 정밀도=0.997, 재현율=0.996, F1=0.997
  Survivor       : 정밀도=0.997, 재현율=0.996, F1=0.997
  Explorer       : 정밀도=0.983, 재현율=0.966, F1=0.975
  Explorer       : 정밀도=0.983, 재현율=0.966, F1=0.975
  Explorer       : 정밀도=0.983, 재현율=0.966, F1=0.975
  Explorer       : 정밀도=0.983, 재현율=0.966, F1=0.975
  Explorer       : 정밀도=0.983, 재현율=0.966, F1=0.975
  Aggressive     : 정밀도=1.000, 재현율=1.000, F1=1.000

📊 전체 평균:
  Macro Avg    : F1=0.987
  Weighted Avg : F1=0.990
  정확도: 0.9899
  F1 (Macro): 0.9874
```

[!ScreenShot](image/Residual모델평가.png)

``` bash
🏆 최고 성능 모델: Basic
  정확도: 0.9925

==================== 5단계: 최고 모델 상세 분석 ====================

🔄 혼동 행렬 생성
----------------------------------------
```

[!ScreenShot](image/5단계_최고모델상세분석.png)

``` bash
📊 클래스별 분류 정확도:
  Survivor       : 0.996 (99.6%)
  Survivor       : 0.999 (99.9%)
  Explorer       : 0.990 (99.0%)
  Explorer       : 0.994 (99.4%)
  Explorer       : 0.969 (96.9%)
  Explorer       : 0.996 (99.6%)
  Explorer       : 0.967 (96.7%)
  Aggressive     : 1.000 (100.0%)

🎯 예측 신뢰도 분석
----------------------------------------
```

[!ScreenShot](image/예측_신뢰도_분석.png)

``` bash
📊 신뢰도 통계:
  평균 신뢰도: 0.990
  신뢰도 표준편차: 0.050
  높은 신뢰도 (>0.8) 비율: 98.2%
  낮은 신뢰도 (<0.5) 비율: 0.0%

🔍 Permutation Importance 특성 중요도 분석
----------------------------------------
🔄 특성별 중요도 계산 중...
  진행: 1/30
  진행: 11/30
  진행: 21/30
```

[!ScreenShot](image/Permutation_Importance특성_중요도_분석.png)

``` bash
📊 상위 10개 중요 특성:
   1. has_kills                : 0.3232
   2. walkDistance_log         : 0.0788
   3. walkDistance             : 0.0751
   4. total_distance           : 0.0634
   5. has_swimDistance         : 0.0609
   6. weaponsAcquired          : 0.0588
   7. killPlace                : 0.0573
   8. damageDealt              : 0.0519
   9. rideDistance             : 0.0512
  10. heal_boost_ratio         : 0.0501

==================== 6단계: 앙상블 모델 ====================

🤝 앙상블 모델 생성 (소프트 보팅)
----------------------------------------
✅ Basic 모델 예측 완료
✅ Advanced 모델 예측 완료
✅ Residual 모델 예측 완료
📊 앙상블 모델 성능:
  정확도: 0.9927

📈 성능 향상 비교:
  vs Basic: +0.0002 (+0.02%)
  vs Advanced: +0.0031 (+0.31%)
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
  vs Residual: +0.0028 (+0.28%)

==================== 딥러닝 파이프라인 완료 요약 ====================
⏰ 실행 시간: 525.6초
🎯 훈련된 모델 수: 3개
🏆 최고 성능 모델: Basic
📊 최고 모델 성능:
  정확도: 0.9925
  F1 (Macro): 0.9867
🤝 앙상블 모델 성능: 0.9927
🚀 앙상블 향상도: +0.0002 (+0.02%)
✅ Phase 6 완료! Phase 7 (모델 해석)로 진행 가능

📊 Phase 6 실행 결과:
최고 모델: Basic
최고 성능: 0.9925
앙상블 성능: 0.9927

💾 모델 저장 중...

💾 최고 성능 모델 저장
----------------------------------------
✅ 모델 저장 완료: pubg_best_model.h5
✅ 메타데이터 저장 완료: pubg_best_model_metadata.json
🔮 예측 함수 생성 중...

🎯 예측 함수 생성
----------------------------------------
✅ 예측 함수 생성 완료
📊 모델 정보:
  특성 수: 30
  클러스터 수: 8
  모델 정확도: 0.9925

🎉 Phase 6 완료!
✅ 다음 단계: Phase 7 - 모델 해석 및 고도화

📝 Phase 6 사용 가능한 함수들:
• dl_results: 딥러닝 결과
• predict_fn: 새 플레이어 예측 함수
• model_path: 저장된 모델 경로
```

## Phase 7: 모델 해석 및 고도화

### 1. 필요한 변수들 확인 및 설정
``` python

print("🔍 Phase 7: 모델 해석 및 고도화 ")
print("="*60)

import warnings
warnings.filterwarnings('ignore')

print("📋 Phase 7 실행 전 변수 확인...")

required_vars = ['dl_results', 'clustering_results', 'modeling_data']
missing_vars = []

for var_name in required_vars:
    if var_name in globals():
        print(f"✅ {var_name}: 존재함")
    else:
        print(f"❌ {var_name}: 없음")
        missing_vars.append(var_name)

if missing_vars:
    print(f"⚠️ 누락된 변수: {missing_vars}")
    print("🔄 Phase 6을 먼저 실행해주세요.")
else:
    print("✅ 모든 필요 변수 준비 완료!")

print("\n" + "="*60)
```
### 2. 간소화된 SHAP 분석 함수
``` python
def simplified_shap_analysis(model, X_sample, feature_names, max_samples=50):
    """간소화된 SHAP 분석 (오류 방지용)"""
    print("🔍 간소화된 SHAP 분석 시작...")

    try:
        # SHAP 설치 확인
        try:
            import shap
            print("✅ SHAP 라이브러리 로드 성공")
        except ImportError:
            print("📦 SHAP 설치 중...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "--quiet"])
            import shap
            print("✅ SHAP 설치 및 로드 완료")

        # 작은 샘플로 분석
        small_sample = X_sample.head(max_samples)
        print(f"🎯 분석 샘플: {len(small_sample)}개")

        # 간단한 배경 데이터 (평균값 사용)
        background = X_sample.mean().values.reshape(1, -1)

        # KernelExplainer 사용 (더 안정적)
        def model_predict(X):
            if hasattr(X, 'values'):
                X = X.values
            return model.predict(X, verbose=0)

        explainer = shap.KernelExplainer(model_predict, background)

        print("🔄 SHAP 값 계산 중... (시간이 소요될 수 있습니다)")
        shap_values = explainer.shap_values(small_sample.values, nsamples=50)

        # 간단한 특성 중요도 계산
        if isinstance(shap_values, list):
            # 다중 클래스인 경우 첫 번째 클래스 사용
            importance_values = np.abs(shap_values[0]).mean(axis=0)
        else:
            importance_values = np.abs(shap_values).mean(axis=0)

        # 특성 중요도 DataFrame 생성
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        print("✅ SHAP 분석 완료")
        return feature_importance_df, shap_values, small_sample

    except Exception as e:
        print(f"⚠️ SHAP 분석 실패: {e}")
        print("🔄 대안 방법으로 특성 중요도 계산...")

        # 대안: Permutation Importance
        from sklearn.metrics import accuracy_score

        # 베이스라인 성능
        y_pred_baseline = np.argmax(model.predict(X_sample.values, verbose=0), axis=1)

        importance_scores = []
        for i, feature_name in enumerate(feature_names):
            # 특성 셞플
            X_shuffled = X_sample.copy()
            X_shuffled.iloc[:, i] = np.random.permutation(X_shuffled.iloc[:, i].values)

            # 성능 측정
            y_pred_shuffled = np.argmax(model.predict(X_shuffled.values, verbose=0), axis=1)

            # 중요도 = 성능 차이
            importance = np.mean(y_pred_baseline != y_pred_shuffled)
            importance_scores.append(importance)

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        print("✅ 대안 특성 중요도 계산 완료")
        return feature_importance_df, None, X_sample.head(20)
```
### 3. 간소화된 비즈니스 인사이트 함수
``` python
def generate_business_insights_simple(cluster_profiles, cluster_names, feature_importance_df):
    """간소화된 비즈니스 인사이트 생성"""
    print("💡 비즈니스 인사이트 생성...")

    insights = {}

    # 클러스터별 특성 분석
    print("🎯 플레이어 유형별 핵심 특징:")
    for cluster_id, profile in cluster_profiles.items():
        cluster_name = cluster_names.get(cluster_id, f'Cluster_{cluster_id}')
        size_pct = profile['percentage']

        print(f"\n📊 {cluster_name} ({size_pct:.1f}% of players)")

        # 상위 특성
        if hasattr(profile.get('top_features', []), 'head'):
            top_features = profile['top_features'].head(3)
            for feature, ratio in top_features.items():
                print(f"  • {feature}: {ratio:.2f}x above average")

        insights[cluster_name] = {
            'percentage': size_pct,
            'cluster_id': cluster_id
        }

    # 전체 게임 밸런스
    cluster_sizes = [profile['percentage'] for profile in cluster_profiles.values()]
    balance_score = np.std(cluster_sizes)

    if balance_score < 5:
        balance_level = "Very Balanced"
    elif balance_score < 10:
        balance_level = "Balanced"
    else:
        balance_level = "Imbalanced"

    insights['game_balance'] = {
        'level': balance_level,
        'score': balance_score
    }

    print(f"\n🎮 게임 밸런스: {balance_level} (편차: {balance_score:.2f})")

    # 핵심 성공 요인
    print(f"\n⭐ 핵심 성공 요인 (상위 5개):")
    for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

    insights['success_factors'] = feature_importance_df.head(5).to_dict('records')

    return insights
```
### 4. 간소화된 이상치 탐지 시스템
``` python
def create_simple_anomaly_detector(clustering_results, modeling_data):
    """간소화된 이상치 탐지 시스템"""
    print("🚨 이상치 탐지 시스템 구축...")

    kmeans_model = clustering_results['kmeans_model']
    X_train = modeling_data['X_train']

    # 각 클러스터의 평균 거리 계산
    cluster_centers = kmeans_model.cluster_centers_
    cluster_labels = clustering_results['cluster_labels']

    # 임계값 계산
    distances = []
    for i, center in enumerate(cluster_centers):
        cluster_data = X_train[cluster_labels == i]
        if len(cluster_data) > 0:
            dist = np.linalg.norm(cluster_data - center, axis=1)
            distances.extend(dist)

    threshold = np.percentile(distances, 95)  # 상위 5%를 이상치로 간주

    def detect_anomaly(player_features):
        """플레이어 이상치 탐지"""
        # 가장 가까운 클러스터 찾기
        distances_to_centers = [np.linalg.norm(player_features - center)
                               for center in cluster_centers]
        min_distance = min(distances_to_centers)
        closest_cluster = np.argmin(distances_to_centers)

        is_anomaly = min_distance > threshold
        anomaly_score = min_distance / threshold

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'closest_cluster': int(closest_cluster),
            'distance': min_distance,
            'threshold': threshold
        }

    print(f"✅ 이상치 탐지 시스템 완료 (임계값: {threshold:.3f})")
    return detect_anomaly
```
### 5. 종합 Phase 7 실행
``` python
def run_phase7_simplified():
    """Phase 7 간소화 실행"""
    print("🚀 Phase 7 간소화 파이프라인 시작")
    print("="*50)

    start_time = time.time()

    try:
        # 최고 모델 추출
        best_model_name = dl_results['best_model_name']
        best_model = dl_results['models'][best_model_name]

        X_test = dl_results['classification_data']['X_test']
        feature_names = dl_results['classification_data']['feature_names']
        cluster_names = dl_results['classification_data']['cluster_names']

        print(f"🎯 분석 대상 모델: {best_model_name}")
        print(f"📊 테스트 데이터: {X_test.shape}")

        # 1. 특성 중요도 분석 (SHAP 또는 대안)
        print(f"\n{'='*15} 1. 특성 중요도 분석 {'='*15}")
        feature_importance_df, shap_values, sample_data = simplified_shap_analysis(
            best_model, X_test.head(100), feature_names
        )

        # 2. 비즈니스 인사이트 생성
        print(f"\n{'='*15} 2. 비즈니스 인사이트 {'='*15}")
        business_insights = generate_business_insights_simple(
            clustering_results['cluster_profiles'],
            clustering_results['cluster_names'],
            feature_importance_df
        )

        # 3. 이상치 탐지 시스템
        print(f"\n{'='*15} 3. 이상치 탐지 시스템 {'='*15}")
        anomaly_detector = create_simple_anomaly_detector(clustering_results, modeling_data)

        # 테스트
        print("🧪 이상치 탐지 테스트...")
        test_samples = X_test.head(5)
        for i, (_, row) in enumerate(test_samples.iterrows(), 1):
            result = anomaly_detector(row.values)
            status = "🚨 이상치" if result['is_anomaly'] else "✅ 정상"
            print(f"  샘플 {i}: {status} (점수: {result['anomaly_score']:.3f})")

        # 4. 간단한 A/B 테스트 설계
        print(f"\n{'='*15} 4. A/B 테스트 프레임워크 {'='*15}")
        ab_test_suggestions = {}

        for cluster_name, insight in business_insights.items():
            if cluster_name != 'game_balance':
                ab_test_suggestions[cluster_name] = [
                    f"{cluster_name} 그룹 대상 맞춤형 컨텐츠 테스트",
                    f"{cluster_name} 플레이어 리워드 시스템 개선",
                    f"{cluster_name} 유형별 튜토리얼 최적화"
                ]

        print("🧪 A/B 테스트 제안:")
        for cluster_name, suggestions in ab_test_suggestions.items():
            print(f"\n🎯 {cluster_name}:")
            for suggestion in suggestions[:2]:
                print(f"  • {suggestion}")

        execution_time = time.time() - start_time

        # 결과 정리
        interpretation_results = {
            'feature_importance': feature_importance_df,
            'business_insights': business_insights,
            'anomaly_detector': anomaly_detector,
            'ab_test_suggestions': ab_test_suggestions,
            'shap_analysis': {
                'success': shap_values is not None,
                'sample_data': sample_data
            }
        }

        print(f"\n{'='*15} Phase 7 완료 요약 {'='*15}")
        print(f"⏰ 실행 시간: {execution_time:.1f}초")
        print(f"🎯 분석된 특성: {len(feature_names)}개")
        print(f"💡 비즈니스 인사이트: {len(business_insights)}개 분야")
        print(f"🧪 A/B 테스트 제안: {len(ab_test_suggestions)}개 그룹")
        print(f"🚨 이상치 탐지: {'활성화' if anomaly_detector else '비활성화'}")

        # 핵심 인사이트 요약
        print(f"\n💎 핵심 인사이트:")
        if 'game_balance' in business_insights:
            balance = business_insights['game_balance']
            print(f"  🎮 게임 밸런스: {balance['level']}")

        top_feature = feature_importance_df.iloc[0]
        print(f"  ⭐ 최고 중요 특성: {top_feature['feature']} ({top_feature['importance']:.4f})")

        print(f"✅ Phase 7 완료! Phase 8 (배포)로 진행 가능")

        return interpretation_results

    except Exception as e:
        print(f"❌ Phase 7 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None
```
### 실행
``` python
if not missing_vars:
    print("🎬 Phase 7 실행 시작!")
    interpretation_results = run_phase7_simplified()

    if interpretation_results:
        print("\n🎉 Phase 7 성공적으로 완료!")
        print("📊 생성된 결과:")
        print(f"  • feature_importance: 특성 중요도 분석")
        print(f"  • business_insights: 비즈니스 인사이트")
        print(f"  • anomaly_detector: 이상치 탐지 함수")
        print(f"  • ab_test_suggestions: A/B 테스트 제안")

        # 이상치 탐지 함수 사용 예제
        print(f"\n🔍 이상치 탐지 함수 사용 예제:")
        print(f"# new_player = [5.0, 350.0, 2000.0, ...]  # 플레이어 특성")
        print(f"# result = interpretation_results['anomaly_detector'](new_player)")
        print(f"# print('이상치 여부:', result['is_anomaly'])")

    else:
        print("❌ Phase 7 실행 실패")
        print("🔄 이전 Phase들을 다시 확인해보세요.")
else:
    print("❌ 필요한 변수가 없어 Phase 7를 실행할 수 없습니다.")
    print("🔄 Phase 6을 먼저 성공적으로 완료해주세요.")
```

#### 실행 결과

``` bash
🎬 Phase 7 실행 시작!
🚀 Phase 7 간소화 파이프라인 시작
==================================================
🎯 분석 대상 모델: Basic
📊 테스트 데이터: (20000, 30)

=============== 1. 특성 중요도 분석 ===============
🔍 간소화된 SHAP 분석 시작...
✅ SHAP 라이브러리 로드 성공
🎯 분석 샘플: 50개
🔄 SHAP 값 계산 중... (시간이 소요될 수 있습니다)
100%
 50/50 [00:11<00:00,  4.98it/s]
⚠️ SHAP 분석 실패: Per-column arrays must each be 1-dimensional
🔄 대안 방법으로 특성 중요도 계산...
✅ 대안 특성 중요도 계산 완료

=============== 2. 비즈니스 인사이트 ===============
💡 비즈니스 인사이트 생성...
🎯 플레이어 유형별 핵심 특징:

📊 Survivor (18.2% of players)
  • heal_boost_ratio: 775.29x above average
  • assists: 479.86x above average
  • has_swimDistance: 294.02x above average

📊 Survivor (31.2% of players)
  • heal_boost_ratio: 1861.49x above average
  • assists: 964.62x above average
  • damage_per_kill: 864.65x above average

📊 Explorer (13.4% of players)
  • walkDistance_log: 3743.08x above average
  • walkDistance: 1179.09x above average
  • revives: 626.25x above average

📊 Explorer (19.9% of players)
  • walkDistance_log: 2245.80x above average
  • longestKill: 610.84x above average
  • has_kills: 501.94x above average

📊 Explorer (5.4% of players)
  • walkDistance_log: 4451.52x above average
  • walkDistance: 1845.04x above average
  • revives: 1551.02x above average

📊 Explorer (5.1% of players)
  • walkDistance_log: 4139.77x above average
  • walkDistance: 1544.91x above average
  • weaponsAcquired: 451.79x above average

📊 Explorer (6.7% of players)
  • walkDistance_log: 3995.30x above average
  • matchDuration: 1400.69x above average
  • walkDistance: 1327.73x above average

📊 Aggressive (0.1% of players)
  • kill_efficiency: 23396.88x above average
  • damage_per_kill: 1435.73x above average
  • assists: 920.03x above average

🎮 게임 밸런스: Balanced (편차: 9.56)

⭐ 핵심 성공 요인 (상위 5개):
  1. has_kills: 0.2700
  2. walkDistance_log: 0.1200
  3. weaponsAcquired: 0.1000
  4. walkDistance: 0.0900
  5. damageDealt_log: 0.0700

=============== 3. 이상치 탐지 시스템 ===============
🚨 이상치 탐지 시스템 구축...
✅ 이상치 탐지 시스템 완료 (임계값: 6.761)
🧪 이상치 탐지 테스트...
  샘플 1: ✅ 정상 (점수: 0.393)
  샘플 2: ✅ 정상 (점수: 0.682)
  샘플 3: ✅ 정상 (점수: 0.344)
  샘플 4: ✅ 정상 (점수: 0.263)
  샘플 5: ✅ 정상 (점수: 0.398)

=============== 4. A/B 테스트 프레임워크 ===============
🧪 A/B 테스트 제안:

🎯 Survivor:
  • Survivor 그룹 대상 맞춤형 컨텐츠 테스트
  • Survivor 플레이어 리워드 시스템 개선

🎯 Explorer:
  • Explorer 그룹 대상 맞춤형 컨텐츠 테스트
  • Explorer 플레이어 리워드 시스템 개선

🎯 Aggressive:
  • Aggressive 그룹 대상 맞춤형 컨텐츠 테스트
  • Aggressive 플레이어 리워드 시스템 개선

🎯 success_factors:
  • success_factors 그룹 대상 맞춤형 컨텐츠 테스트
  • success_factors 플레이어 리워드 시스템 개선

=============== Phase 7 완료 요약 ===============
⏰ 실행 시간: 17.1초
🎯 분석된 특성: 30개
💡 비즈니스 인사이트: 5개 분야
🧪 A/B 테스트 제안: 4개 그룹
🚨 이상치 탐지: 활성화

💎 핵심 인사이트:
  🎮 게임 밸런스: Balanced
  ⭐ 최고 중요 특성: has_kills (0.2700)
✅ Phase 7 완료! Phase 8 (배포)로 진행 가능

🎉 Phase 7 성공적으로 완료!
📊 생성된 결과:
  • feature_importance: 특성 중요도 분석
  • business_insights: 비즈니스 인사이트
  • anomaly_detector: 이상치 탐지 함수
  • ab_test_suggestions: A/B 테스트 제안

🔍 이상치 탐지 함수 사용 예제:
# new_player = [5.0, 350.0, 2000.0, ...]  # 플레이어 특성
# result = interpretation_results['anomaly_detector'](new_player)
# print('이상치 여부:', result['is_anomaly'])
```