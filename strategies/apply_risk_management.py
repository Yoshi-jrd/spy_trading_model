def apply_risk_management(profit_loss, trade_outcome, risk_management):
    """
    Applies risk management rules like stop-loss or take-profit to adjust the profit/loss
    and determine if the trade outcome is successful or not.
    """
    stop_loss = risk_management.get('stop_loss')
    take_profit = risk_management.get('take_profit')

    # Adjust profit/loss based on stop-loss or take-profit
    if profit_loss <= stop_loss:
        trade_outcome = False  # Trade failed, stop-loss hit
        return stop_loss, trade_outcome  # Close trade at stop-loss
    
    if profit_loss >= take_profit:
        trade_outcome = True  # Trade succeeded, take-profit hit
        return take_profit, trade_outcome  # Close trade at take-profit

    # If neither stop-loss nor take-profit hit, return the original values
    return profit_loss, trade_outcome if trade_outcome is not None else profit_loss > 0
