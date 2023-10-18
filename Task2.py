import pandas as pd

def final_fixed_transfer_contract_value(injection_dates, withdrawal_dates, gas_prices, 
                                        transfer_rate, maximum_volume, storage_cost_per_month):
    """
    Calculates the value of the contract based on the provided parameters.
    
    Parameters:
    - injection_dates (list): List of dates (days) when gas is injected into storage.
    - withdrawal_dates (list): List of dates (days) when gas is withdrawn from storage.
    - gas_prices (DataFrame): Dataframe of gas prices on the given date ($).
    - transfer_rate (float): Rate of gas to that can be injected/withdrawn per day (MMBtu/day).
    - maximum_volume (float): Maximum storage capacity (MMBtu).
    - storage_cost_per_month (float): Cost to store gas per month ($/month).

    Returns:
    (float): Returns the value of the contract ($).
    """
    # Convert all dates to datetime format
    injection_dates = pd.to_datetime(injection_dates)
    withdrawal_dates = pd.to_datetime(withdrawal_dates)
    gas_prices['Dates'] = pd.to_datetime(gas_prices['Dates'])

    # Ensure storage capacity is not exceeded
    injection_volume = len(injection_dates) * transfer_rate
    if injection_volume > maximum_volume:
        raise ValueError('The total volume injected exceeds the storage capacity.')
    
    # Initialise revenues
    sales_revenue = 0

    # Calculate the gas sales revenue
    for date in withdrawal_dates:
        price = gas_prices[gas_prices['Dates'] == date]['Prices'].iloc[0]
        sales_revenue += price * transfer_rate
    
    # Total revenue 
    total_revenue = sales_revenue

    # Initialise costs
    transfer_cost = 0
    purchase_cost = 0
    storage_cost = 0

    # Calculate the cost to withdraw and inject the gas (Fixed cost per transfer event)
    transfer_dates = injection_dates.tolist() + withdrawal_dates.tolist()
    cost_per_transfer = 10000  # Fixed cost per transfer event
    transfer_cost = cost_per_transfer * len(transfer_dates)

    # Calculate the gas purchase cost
    for date in injection_dates:
        cost = gas_prices[gas_prices['Dates'] == date]['Prices'].iloc[0]
        purchase_cost += cost * transfer_rate

    # Sort the dates for consistent processing
    injection_dates = sorted(injection_dates)
    withdrawal_dates = sorted(withdrawal_dates)

    # Calculate total storage months
    total_storage_months = 0
    for date_i in injection_dates:
        for date_w in withdrawal_dates:
            if date_i < date_w:
                months_difference = (date_w.year - date_i.year) * 12 + (date_w.month - date_i.month)
                total_storage_months += months_difference
                break

    # Calculate the total storage cost
    storage_cost = total_storage_months * storage_cost_per_month

    # Total costs from purchase, transfer, and storage
    total_cost = purchase_cost + storage_cost + transfer_cost

    # Calculate the contract's net value
    contract_value_result = total_revenue - total_cost

    return contract_value_result
