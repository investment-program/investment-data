from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import Optional, List
from datetime import datetime

from config import Config
from database import Base, StockAnalysis


class DBViewer:
    """SQLAlchemy를 사용한 데이터베이스 조회 클래스"""

    def __init__(self, config: Config):
        """
        Parameters:
            config (Config): 설정 객체
        """
        self.config = config
        self.engine = None
        self.Session = None

    def connect(self):
        """데이터베이스 연결을 생성합니다."""
        try:
            self.engine = create_engine(f"sqlite:///{self.config.file.DB_PATH}")
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            print("데이터베이스 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 실패: {str(e)}")

    def close(self):
        """데이터베이스 연결을 종료합니다."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None

    def get_all_stock_names(self) -> Optional[List[str]]:
        """
        데이터베이스에 저장된 모든 종목명을 조회합니다.

        Returns:
            Optional[List[str]]: 종목명 리스트 또는 실패시 None
            예시: ['삼성전자', 'SK하이닉스', 'NAVER', ...]

        Note:
            - 종목명은 한글명으로 반환됩니다.
            - 알파벳 순으로 정렬됩니다.
            - 중복된 종목명은 제거됩니다.
        """
        if not self.Session:
            self.connect()

        try:
            session = self.Session()
            # name 컬럼만 조회하고 중복 제거 후 정렬
            stocks = (
                session.query(StockAnalysis.name)
                .distinct()
                .order_by(StockAnalysis.name)
                .all()
            )

            # 튜플 리스트를 문자열 리스트로 변환
            stock_names = [stock[0] for stock in stocks]

            print(f"총 {len(stock_names)}개 종목 조회 완료")
            return stock_names

        except Exception as e:
            print(f"종목명 조회 실패: {str(e)}")
            return None
        finally:
            if session:
                session.close()

    def get_all_stocks(self) -> Optional[pd.DataFrame]:
        """
        모든 종목의 분석 데이터를 조회합니다.

        Returns:
            Optional[pd.DataFrame]: 전체 종목 데이터 또는 None
        """
        if not self.Session:
            self.connect()

        session = None
        try:
            session = self.Session()
            stocks = (
                session.query(StockAnalysis)
                .order_by(desc(StockAnalysis.dividend_yield))
                .all()
            )

            data = []
            for stock in stocks:
                data.append(
                    {
                        "종목코드": stock.code,
                        "종목명": stock.name,
                        "연간수익률(%)": round(stock.annual_return * 100, 2),
                        "변동성(%)": round(stock.volatility * 100, 2),
                        "샤프비율": round(stock.sharpe_ratio, 2),
                        "일평균거래대금(백만원)": round(stock.liquidity / 1_000_000, 0),
                        "배당수익률(%)": round(stock.dividend_yield, 2),
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            print(f"데이터 조회 실패: {str(e)}")
            return None
        finally:
            if session:
                session.close()

    def get_top_dividend_stocks(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        배당수익률 상위 종목을 조회합니다.

        Parameters:
            limit (int): 조회할 종목 수

        Returns:
            Optional[pd.DataFrame]: 상위 종목 데이터 또는 None
        """
        if not self.Session:
            self.connect()

        try:
            session = self.Session()
            stocks = (
                session.query(StockAnalysis)
                .order_by(desc(StockAnalysis.dividend_yield))
                .limit(limit)
                .all()
            )

            data = []
            for stock in stocks:
                data.append(
                    {
                        "종목코드": stock.code,
                        "종목명": stock.name,
                        "배당수익률(%)": round(stock.dividend_yield, 2),
                        "연간수익률(%)": round(stock.annual_return * 100, 2),
                        "변동성(%)": round(stock.volatility * 100, 2),
                        "샤프비율": round(stock.sharpe_ratio, 2),
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            print(f"데이터 조회 실패: {str(e)}")
            return None
        finally:
            session.close()

    def get_stock_by_conditions(
        self,
        min_dividend: float = 0,
        max_volatility: float = 100,
        min_liquidity: float = 0,
    ) -> Optional[pd.DataFrame]:
        """
        조건에 맞는 종목을 조회합니다.

        Parameters:
            min_dividend (float): 최소 배당수익률
            max_volatility (float): 최대 변동성
            min_liquidity (float): 최소 일평균거래대금(백만원)

        Returns:
            Optional[pd.DataFrame]: 조건에 맞는 종목 데이터 또는 None
        """
        if not self.Session:
            self.connect()

        try:
            session = self.Session()
            stocks = (
                session.query(StockAnalysis)
                .filter(
                    StockAnalysis.dividend_yield >= min_dividend,
                    StockAnalysis.volatility * 100 <= max_volatility,
                    StockAnalysis.liquidity >= min_liquidity * 1_000_000,
                )
                .order_by(desc(StockAnalysis.dividend_yield))
                .all()
            )

            data = []
            for stock in stocks:
                data.append(
                    {
                        "종목코드": stock.code,
                        "종목명": stock.name,
                        "배당수익률(%)": round(stock.dividend_yield, 2),
                        "연간수익률(%)": round(stock.annual_return * 100, 2),
                        "변동성(%)": round(stock.volatility * 100, 2),
                        "샤프비율": round(stock.sharpe_ratio, 2),
                        "일평균거래대금(백만원)": round(stock.liquidity / 1_000_000, 0),
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            print(f"데이터 조회 실패: {str(e)}")
            return None
        finally:
            session.close()


def main():
    """
    종목명 조회 예시
    """
    config = Config()
    viewer = DBViewer(config)

    try:
        print("\n=== 전체 종목명 조회 ===")
        stock_names = viewer.get_all_stock_names()

        if stock_names:
            print(f"\n총 {len(stock_names)}개 종목:")
            # 한 줄에 5개씩 출력
            for i in range(0, len(stock_names), 5):
                chunk = stock_names[i : i + 5]
                print(", ".join(chunk))

    finally:
        viewer.close()


if __name__ == "__main__":
    main()
