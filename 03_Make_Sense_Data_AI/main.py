from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).parent / "data" / "sales.csv"

def main():
    df = pd.read_csv(DATA)
    # Data preparation: buang baris penting yang kosong dan duplikat
    df = df.drop_duplicates().dropna(subset=["tanggal", "produk", "jumlah", "harga"])
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal"])
    df["total"] = df["jumlah"] * df["harga"]

    # 1. Descriptive: apa yang terjadi?
    print("TOTAL PENJUALAN:", f"Rp{df['total'].sum():,.0f}")
    print("\nPenjualan per produk:")
    print(df.groupby("produk")["total"].sum().sort_values(ascending=False))

    # 2. Diagnostic: kelompokkan penjualan menurut hari untuk melihat pola
    daily = df.groupby(df["tanggal"].dt.day)["total"].sum().reset_index()
    daily.columns = ["hari", "total"]
    print("\nPenjualan per hari:")
    print(daily.to_string(index=False))

    # 3. Predictive: regresi sederhana pada urutan hari
    if len(daily) >= 2:
        model = LinearRegression().fit(daily[["hari"]], daily["total"])
        next_day = int(daily["hari"].max()) + 1
        prediction = max(0, model.predict(pd.DataFrame({"hari": [next_day]}))[0])
        print(f"\nPrediksi ilustratif untuk hari ke-{next_day}: Rp{prediction:,.0f}")
    else:
        print("\nData belum cukup untuk membuat model prediksi.")

    # 4. Prescriptive: rekomendasi sederhana berdasarkan produk terlaris
    best = df.groupby("produk")["jumlah"].sum().idxmax()
    print(f"Rekomendasi: periksa ketersediaan stok produk '{best}' karena jumlah terjual tertinggi.")
    print("Catatan: keputusan akhir tetap perlu mempertimbangkan stok aktual dan konteks bisnis.")

if __name__ == "__main__":
    main()
