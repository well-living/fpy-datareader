# fpy-datareader

## 政府統計の総合窓口（e-Stat）のAPI3.0版でデータ取得するPythonコード

- 政府統計の総合窓口（e-Stat）のAPI3.0版の仕様

https://www.e-stat.go.jp/api/api-info/e-stat-manual3-0

### e-StatAPIで統計データ取得

```bash
pip install fpy-datareader
```

```Python
import fpy_datareader as fdr

api_key = "xxxx"
dfs = fdr.get_data_estat_statsdata(api_key, statsDataId="0003109558")
```

```Python
import fpy_datareader.data as web

api_key = "xxxx"
f = web.DataReader("0003109558", "estat", api_key=api_key)
```

```Python
from fpy_datareader import estat

statsdata = estat.StatsDataReader(api_key, statsDataId="0003109558")
df = statsdata.read()
```

### e-StatAPIで統計表情報取得
```Python
import fpy_datareader as fdr

api_key = "xxxx"
statslist = fdr.get_data_estat_statslist(api_key)
```

## クレジット
このサービスは、政府統計総合窓口(e-Stat)のAPI機能を使用していますが、サービスの内容は国によって保証されたものではありません。
https://www.e-stat.go.jp/api/api-info/credit
