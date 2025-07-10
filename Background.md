# Power Markets Background

All European countries have a power grid, which provides electricity for 
everything from consumers’ household appliances to the heavy machinery in 
large factories. The sum total of all the consumption of the power grid 
forms the *load* (often also known as the *demand*). This power demand is 
met by a variety of generators, consisting of everything from solar farms 
and wind turbines to nuclear power stations and gas-fired thermal power 
stations. The sum total of all the power generation that goes onto the grid 
is called the *supply*.

The supply and demand must always balance, otherwise there are severe risks 
of blackouts, and damage to the devices connected to the grid. The 
electricity market exists to help balance supply and demand. Typically, most 
supply and demand cannot be varied instantaneously, and so the market is 
structured to provide a fixed price over a given time window. Each time 
window is called a *market time unit* (MTU). Many EU countries have 
electricity markets which divide the day into 15-minute MTUs (and all EU 
countries are on a roadmap towards this).

The entire power grid in Europe is interconnected, meaning that it is in 
theory possible to route power from one point to any other. However if you 
pick two arbitrary points within the power grid, it’s unlikely that an 
arbitrary amount of power could be routed between them. The market is 
divided into regions called *delivery areas*, and within a delivery area one 
can route an arbitrary amount of power from one point to another. Many 
European countries each have exactly one delivery area (e.g. Great Britain, 
France, Netherlands, Belgium), but others (in particular, the Nordic 
countries) have more than one (Denmark has two, Norway has five, etc).

A substantial proportion of a delivery area’s power is traded in the 
*day-ahead* market. The day-ahead market is a set of auctions that happen in 
the morning on the calendar day before the power is due to be delivered into 
the grid. There is one auction per (delivery area, MTU) pair, and the 
majority of the auctions run at the same time. The auctions can be 
treated as independent. They are structured so 
that there is one final price, and that final price is chosen so that the 
overall amount of power traded is largest. The auction is pay-as-clear, 
which effectively allows people to bid at their marginal cost of generation 
(or marginal value of consumption), and if their bids are met, then they 
will be met at a price at least as good as the one they entered. 
This means that there is exactly one price (both for buyers and sellers) 
for each (delivery area, MTU) pair in the day-ahead market.

In this exercise, we’re going to try to predict the day-ahead power price 
given the forecast wind and solar generation, and the forecast load, each 
given per (delivery area, MTU) pair.

## Data
The data files in the `data/` directory should be fairly self-explanatory.
You can load the data using the `load_data()` helper in `data.py`.

The data contains the following columns: 
- `DateTime`: start of the MTU in question
- `ResolutionCode`: contains the MTU duration in the respective area
- `AreaCode`: unique code of the delivery area
- `Price[Currency/MWh]`
- `TotalLoadValue`: forecast grid load (i.e. energy demand) 
- `Solar`: forecast solar generation
- `Wind Offshore`: forecast offshore wind generation
- `Wind Onshore`: forecast onshore wind generation

## Notes

* Prices are generally positive, but can be negative. Negative prices 
  generally happen due to one of two reasons:
  1. Certain power plants cannot be switched off briefly. If they are going 
     to be switched off, then they have a long cool-down time (and likely a 
     similarly long warm-up time). These plants typically place block bids 
     which cover multiple MTUs. When the auction process runs this can 
     result in negative prices in some MTUs (as long as sufficiently large 
     prices are offered in the neighbouring ones).
  2. A power plant is being paid subsidies according to how much it 
     generates, so as long as the subsidy is enough to cover the negative 
     sale price then it still makes financial sense to operate it.  
* Different delivery areas within the same country often have quite a lot of 
  transmission capacity between them, which means that the prices can quite 
  often be the same.
* Wind and solar have a near-zero marginal cost of generation. You’d expect 
  power to be cheaper when there’s a higher proportion of wind and solar 
  generation, as they undercut the most expensive thermal (coal/gas) plants.  
* The load is measured at the transmission infrastructure. This 
  means that generation that’s directly connected to consumption (called 
  *behind-the-meter* generation) will not be seen in the load value.
