import yfinance as yf
import numpy as np

def basic_dcf_model_flexible(ticker):
    stock = yf.Ticker(ticker)

    try:
        financials = stock.financials
        cashflow = stock.cashflow
        balance_sheet = stock.balance_sheet
        info = stock.info
    except Exception as e:
        return f"❌ Failed to retrieve data: {e}"

    try:
        # Dynamically find Capital Expenditures
        capex = None
        for key in cashflow.index:
            if "capital expend" in key.lower():
                capex = cashflow.loc[key]
                break

        # Dynamically find Operating Cash Flow
        operating_cash_flow = None
        for key in cashflow.index:
            if "total cash from operating activities" in key.lower():
                operating_cash_flow = cashflow.loc[key]
                break

        total_debt = (
            balance_sheet.loc["Long Term Debt"].fillna(0) +
            balance_sheet.loc["Short Long Term Debt"].fillna(0)
            if "Short Long Term Debt" in balance_sheet.index and "Long Term Debt" in balance_sheet.index
            else 0
        )
        cash = balance_sheet.loc["Cash"].fillna(0) if "Cash" in balance_sheet.index else 0
        shares_outstanding = info.get("sharesOutstanding", np.nan)

        if capex is None or operating_cash_flow is None:
            return "⚠️ Required financial fields missing for this ticker. Try another."

        # Estimate average FCF
        fcf = operating_cash_flow.mean() - capex.mean()

        # Forecast 5 years of FCF with 10% growth
        fcf_forecast = [fcf * (1.10 ** i) for i in range(1, 6)]

        discount_rate = 0.10
        terminal_growth = 0.03

        # Terminal value
        terminal_value = fcf_forecast[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # DCF value calculation
        dcf_value = sum([fcf_forecast[i] / (1 + discount_rate) ** (i + 1) for i in range(5)])
        dcf_value += terminal_value / (1 + discount_rate) ** 5

        # Adjust for debt and cash
        firm_value = dcf_value - total_debt + cash
        fair_value_per_share = firm_value / shares_outstanding

        return {
            "Ticker": ticker.upper(),
            "Estimated Fair Value per Share": round(fair_value_per_share, 2),
            "Assumed FCF Growth Rate": "10%",
            "Discount Rate": "10%",
            "Terminal Growth Rate": "3%",
            "DCF Value (Firm)": round(firm_value, 2),
            "Shares Outstanding": shares_outstanding
        }

    except Exception as e:
        return f"❌ Error during DCF calculation: {e}"

if __name__ == "__main__":
    ticker_input = input("Enter a stock ticker (e.g., AAPL, MSFT, TSLA): ").upper()
    result = basic_dcf_model_flexible(ticker_input)

    if isinstance(result, dict):
        print("\n🔍 DCF Valuation Result:")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print(result)
