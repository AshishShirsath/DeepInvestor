import streamlit as st

def calculate_bond_return(principal, rate, months, compounding=False):
    if compounding:
        total_return = principal * ((1 + (rate / (100 * 12))) ** months)
    else:
        total_return = principal + (principal * rate * (months / 12) / 100)
    
    return total_return

st.set_page_config(page_title="Bond Investment Calculator", page_icon="💰", layout="centered")

st.title("💰 Bond Investment Calculator")
st.markdown("### Calculate your bond returns with Simple or Compound Interest.")

P = st.number_input("💵 Enter Investment Amount ($):", min_value=0.0, value=10000.0, step=100.0)
R = st.number_input("📈 Enter Annual Interest Rate (%):", min_value=0.0, value=5.0, step=0.1)
T = st.number_input("📅 Enter Investment Duration (Months):", min_value=1, value=6, step=1)
compounding = st.radio("🔄 Is Interest Compounded Monthly?", ["No", "Yes"])

if st.button("Calculate Return"):
    is_compounded = True if compounding == "Yes" else False
    final_amount = calculate_bond_return(P, R, T, is_compounded)

    st.subheader(f"📊 Investment Summary:")
    st.write(f"💰 **Initial Investment:** ${P:,.2f}")
    st.write(f"📈 **Interest Rate:** {R:.2f}% per year")
    st.write(f"📅 **Investment Period:** {T} months")
    st.write(f"🔄 **Compounding:** {'Yes' if is_compounded else 'No'}")

    st.success(f"🎯 **Total Amount After {T} Months:** **${final_amount:,.2f}**")

st.markdown("---")
st.markdown("🔹 *This tool helps estimate bond returns based on simple or compound interest calculations.*")
