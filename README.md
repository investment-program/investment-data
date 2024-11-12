백테스트는 backtest_main.py 에 run_backtest()로 실행돼요! (나머지는 데이터수집분석+시각화 함수들)

백테스트 리턴값  
```json
{
    "metrics": {
        "period": {
            "start_date": str,    # 예: "2020-01-01"
            "end_date": str       # 예: "2023-12-31"
        },
        "portfolio": {
            "composition": [      # 포트폴리오 구성 종목들의 리스트
                {
                    "code": str,           # 종목코드
                    "name": str,           # 종목명
                    "weight": float,       # 비중 (%)
                    "dividend_yield": float # 배당수익률 (%)
                },
                # ... 다른 종목들
            ],
            "final_value": float,       # 최종 포트폴리오 가치 (원)
            "total_return": float,      # 총 수익률 (%)
            "annual_volatility": float, # 연간 변동성 (%)
            "sharpe_ratio": float,      # 샤프 비율
            "max_drawdown": float,      # 최대 낙폭 (%)
            "win_rate": float          # 승률 (%)
        },
        "benchmark": {
            "final_value": float,       # KOSPI 최종 가치 (원)
            "total_return": float,      # KOSPI 총 수익률 (%)
            "annual_volatility": float  # KOSPI 연간 변동성 (%)
        },
        "individual_stocks": [          # 개별 종목 성과 리스트
            {
                "code": str,         # 종목코드
                "name": str,         # 종목명
                "return": float,     # 수익률 (%)
                "volatility": float  # 변동성 (%)
            },
            # ... 다른 종목들
        ]
    },
    "visualizations": {
        "value_changes": str,     # Base64 인코딩된 포트폴리오 가치 변화 그래프
        "composition": str,       # Base64 인코딩된 포트폴리오 구성 파이 차트
        "risk_return": str        # Base64 인코딩된 위험-수익 산점도
    }
}
```


웹에서 시각화자료 사용하는법
<img src="data:image/png;base64,{visualizations['value_changes']}" />
