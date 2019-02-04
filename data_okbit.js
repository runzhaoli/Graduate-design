/*backtest
start: 2017-11-03 02:00:00
end: 2018-04-05 02:00:00
period: 1h
exchanges: [{"eid":"Bitfinex","currency":"BTC"}]
exchanges: [{"eid":"OKcoinUSD","currency":"BTC"}]
*/


function main(){
	var lastBar = null
    var records = _C(exchange.GetRecords)
    lastBar = records[records.length - 1]
	while(true){
        records = exchange.GetRecords()
        //因为最后一根ticker还没过完，数据不完整，所以records.length - 2
        if(records && lastBar.Time != records[records.length - 1].Time){
            Log(records[records.length - 2].Close)
            //Log(records[records.length - 2].Volume)
            lastBar = records[records.length - 1]
        }
    
		Sleep(500)
	}
}