import pandas as pd
from src.pipelines import data_pipeline


def test_data_pipeline(monkeypatch):

    # FRED yerine kullanacağımız örnek veri.
    fake_df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01",
            "2026-03-01"
        ]),
        "sales": [100, 120, 140]
    })

    # Gerçek veri çekme fonksiyonunu geçici olarak değiştiriyoruz.
    monkeypatch.setattr(
        data_pipeline,
        "load_sales_data",
        lambda: fake_df
    )

    # Pipeline'ı çalıştırıyoruz.
    result = data_pipeline.prepare_sales_data()

    # Sonuçları doğruluyoruz.
    assert not result.empty
    assert "sales" in result.columns
    assert len(result) == 3