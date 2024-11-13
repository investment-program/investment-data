from typing import Tuple, Optional, Dict, List
from backtest.portfolio import Portfolio
from backtest.data_loader import DataLoader
from backtest.optimizer import PortfolioOptimizer
from backtest.backtest_engine import BacktestEngine
from backtest.visualizer import BacktestVisualizer
import sqlite3
import pandas as pd


def run_backtest(
    # 기본 설정
    initial_capital: float = 100_000_000,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    # 종목 선정 조건
    n_stocks: int = 5,
    min_dividend: float = 2.0,  # 최소 배당수익률
    min_liquidity: float = 100,  # 최소 일평균 거래대금 (백만원)
    max_volatility: float = float("inf"),  # 최대 변동성
    # 포트폴리오 최적화 조건
    min_weight: float = 0.05,  # 최소 투자 비중
    max_weight: float = 0.90,  # 최대 투자 비중
    target_return: float = 0.05,  # 목표 수익률
    risk_free_rate: float = 0.03,  # 무위험 수익률
    # 기타 설정
    db_path: str = "data/stock_data.db",
) -> Dict:
    """백테스트 실행 함수

    Returns:
        Dict: 백테스트 결과를 담은 딕셔너리
        {
            "metrics": {
                "period": {
                    "start_date": str,
                    "end_date": str
                },
                "portfolio": {
                    "composition": List[Dict],  # 포트폴리오 구성 종목 정보
                    "final_value": float,       # 최종 포트폴리오 가치
                    "total_return": float,      # 총 수익률
                    "annual_volatility": float, # 연간 변동성
                    "sharpe_ratio": float,      # 샤프 비율
                    "max_drawdown": float,      # 최대 낙폭
                    "win_rate": float          # 승률
                },
                "benchmark": {
                    "final_value": float,
                    "total_return": float,
                    "annual_volatility": float
                },
                "individual_stocks": List[Dict]  # 개별 종목 성과
            },
            "visualizations": {
                "value_changes": str,     # Base64 인코딩된 포트폴리오 가치 변화 그래프
                "composition": str,       # Base64 인코딩된 포트폴리오 구성 파이 차트
                "risk_return": str        # Base64 인코딩된 위험-수익 산점도
            }
        }
    """
    try:
        # 1. 포트폴리오 초기화
        portfolio = Portfolio(
            initial_capital=initial_capital, start_date=start_date, end_date=end_date
        )

        # 2. 데이터 로드
        data_loader = DataLoader(db_path)
        data_loader.load_stock_data(
            portfolio=portfolio,
            n_stocks=n_stocks,
            min_dividend=min_dividend,
            min_liquidity=min_liquidity,
            max_volatility=max_volatility,
        )

        # 3. 포트폴리오 최적화
        optimizer = PortfolioOptimizer(
            min_weight=min_weight,
            max_weight=max_weight,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
        )
        optimizer.optimize(portfolio)

        # 4. 백테스트 실행
        engine = BacktestEngine(portfolio)
        backtest_results = engine.run()

        # 5. 결과 생성
        visualizer = BacktestVisualizer(portfolio)
        results = visualizer.generate_results(backtest_results)

        return results

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return {"error": str(e), "metrics": None, "visualizations": None}


def specific_backtest(
    stock_names: List[str],
    # 기본 설정
    initial_capital: float = 100_000_000,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    # 포트폴리오 최적화 조건
    min_weight: float = 0.05,  # 최소 투자 비중
    max_weight: float = 0.90,  # 최대 투자 비중
    target_return: float = 0.05,  # 목표 수익률
    risk_free_rate: float = 0.03,  # 무위험 수익률
    # 기타 설정
    db_path: str = "data/stock_data.db",
) -> Dict:
    """특정 종목들로 구성된 포트폴리오의 백테스트를 수행하는 함수

    Returns:
        Dict: 백테스트 결과를 담은 딕셔너리
        {
            "metrics": {
                "period": {
                    "start_date": str,
                    "end_date": str
                },
                "portfolio": {
                    "composition": List[Dict],  # 포트폴리오 구성 종목 정보
                    "final_value": float,       # 최종 포트폴리오 가치
                    "total_return": float,      # 총 수익률
                    "annual_volatility": float, # 연간 변동성
                    "sharpe_ratio": float,      # 샤프 비율
                    "max_drawdown": float,      # 최대 낙폭
                    "win_rate": float          # 승률
                },
                "benchmark": {
                    "final_value": float,
                    "total_return": float,
                    "annual_volatility": float
                },
                "individual_stocks": List[Dict]  # 개별 종목 성과
            },
            "visualizations": {
                "value_changes": str,     # Base64 인코딩된 포트폴리오 가치 변화 그래프
                "composition": str,       # Base64 인코딩된 포트폴리오 구성 파이 차트
                "risk_return": str        # Base64 인코딩된 위험-수익 산점도
            }
        }

    Parameters:
        stock_names (List[str]): 백테스트할 종목명 리스트
        initial_capital (float, optional): 초기 투자금액. Defaults to 100_000_000.
        start_date (str, optional): 백테스트 시작일. Defaults to "2020-01-01".
        end_date (str, optional): 백테스트 종료일. Defaults to "2023-12-31".
        min_weight (float, optional): 최소 투자 비중. Defaults to 0.05.
        max_weight (float, optional): 최대 투자 비중. Defaults to 0.40.
        target_return (float, optional): 목표 수익률. Defaults to 0.05.
        risk_free_rate (float, optional): 무위험 수익률. Defaults to 0.03.
        db_path (str, optional): DB 파일 경로. Defaults to "data/stock_data.db".
    """
    try:
        # 1. 포트폴리오 초기화
        portfolio = Portfolio(
            initial_capital=initial_capital, start_date=start_date, end_date=end_date
        )

        # 2. 데이터 로드
        data_loader = DataLoader(db_path)

        # 종목명으로 종목 코드 조회
        with sqlite3.connect(db_path) as conn:
            placeholders = ",".join(["?" for _ in stock_names])
            query = f"""
            SELECT code, name, dividend_yield, liquidity
            FROM stock_analysis
            WHERE name IN ({placeholders})
            """
            stock_data = pd.read_sql(query, conn, params=stock_names)

        if len(stock_data) != len(stock_names):
            missing_stocks = set(stock_names) - set(stock_data["name"])
            raise ValueError(f"다음 종목들을 찾을 수 없습니다: {missing_stocks}")

        # 3. 주가 데이터 로드
        data_loader.load_stock_data(
            portfolio=portfolio, stock_codes=stock_data["code"].tolist()
        )

        # 4. 포트폴리오 최적화
        optimizer = PortfolioOptimizer(
            min_weight=min_weight,
            max_weight=max_weight,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
        )
        optimizer.optimize(portfolio)

        # 5. 백테스트 실행
        engine = BacktestEngine(portfolio)
        backtest_results = engine.run()

        # 6. 결과 생성
        visualizer = BacktestVisualizer(portfolio)
        results = visualizer.generate_results(backtest_results)

        return results

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return {"error": str(e), "metrics": None, "visualizations": None}


if __name__ == "__main__":
    # 테스트를 위한 실행
    results = specific_backtest(
        stock_names=["삼성전자", "SK하이닉스", "현대차", "NAVER"]
    )

    # 결과 확인
    if results.get("metrics"):
        print("\n=== 백테스트 결과 ===")
        print(
            f"포트폴리오 총 수익률: {results['metrics']['portfolio']['total_return']:.2f}%"
        )
        print(
            f"벤치마크 총 수익률: {results['metrics']['benchmark']['total_return']:.2f}%"
        )
        print(f"샤프 비율: {results['metrics']['portfolio']['sharpe_ratio']:.2f}")
        print(f"최대 낙폭: {results['metrics']['portfolio']['max_drawdown']:.2f}%")

        print("\n포트폴리오 구성:")
        for stock in results["metrics"]["portfolio"]["composition"]:
            print(
                f"{stock['name']}: {stock['weight']:.1f}% "
                f"(배당수익률: {stock['dividend_yield']:.1f}%)"
            )
