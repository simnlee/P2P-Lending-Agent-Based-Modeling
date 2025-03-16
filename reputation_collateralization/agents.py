from mesa import Agent
import numpy as np

class LenderAgent(Agent):
    """An agent that provides liquidity to the lending pool."""
    
    def __init__(self, unique_id, model):
        super().__init__(model)
        # Initial capital to provide as liquidity
        self.capital = self.random.uniform(1000, 10000)
        # Amount currently supplied to the protocol
        self.supplied_amount = 0
        # Accumulated interest from lending
        self.accumulated_interest = 0
        # Supply initial capital
        initial_supply = self.capital * 0.7  # Supply 70% of capital initially
        self.supplied_amount = initial_supply
        self.capital -= initial_supply
        self.model.add_liquidity(initial_supply)
        
    def step(self):
        """
        At each step, the lender:
        1. Decides whether to supply more capital
        2. Collects interest from existing supplied amount
        """
        # Supply more capital if utilization rate is high and we have capital
        utilization_rate = self.model.get_utilization_rate()
        if utilization_rate > 0.7 and self.capital > 0:
            supply_amount = min(self.capital, self.capital * 0.5)
            self.supplied_amount += supply_amount
            self.capital -= supply_amount
            self.model.add_liquidity(supply_amount)
        
        # Withdraw some capital if utilization rate is too low
        elif utilization_rate < 0.3 and self.supplied_amount > 0:
            withdraw_amount = self.supplied_amount * 0.1
            self.supplied_amount -= withdraw_amount
            self.capital += withdraw_amount
            self.model.remove_liquidity(withdraw_amount)
            
        # Collect interest
        interest = self.model.calculate_lender_interest(self.supplied_amount)
        self.accumulated_interest += interest

class BorrowerAgent(Agent):
    """An agent that borrows from the lending pool."""
    
    def __init__(self, unique_id, model):
        super().__init__(model)
        # Initial collateral the borrower has
        self.collateral = self.random.uniform(500, 5000)
        # Amount currently borrowed
        self.borrowed_amount = 0
        # Reputation score (0 to 1)
        self.reputation_score = 0.5
        # History of successful repayments
        self.successful_repayments = 0
        # History of defaults
        self.defaults = 0
        # Whether the agent is currently liquidated
        self.is_liquidated = False
        # Time since last borrow (to prevent immediate reborrowing)
        self.time_since_last_borrow = 0
        
    def get_max_borrow_amount(self):
        """Calculate maximum amount that can be borrowed based on collateral and reputation."""
        base_collateral_factor = self.model.base_collateral_factor
        
        # Reputation bonus: higher reputation = higher collateral factor (can borrow more with same collateral)
        if self.model.reputation_sensitivity == 0:
            # Control case: more conservative collateral factor
            adjusted_collateral_factor = base_collateral_factor * 0.8  # 20% more conservative
        else:
            # Higher reputation allows borrowing more with the same collateral
            reputation_bonus = self.model.reputation_sensitivity * (self.reputation_score - 0.5) * 0.3
            adjusted_collateral_factor = min(0.95, max(0.3, base_collateral_factor + reputation_bonus))
        
        # Calculate max borrow amount
        return self.collateral * adjusted_collateral_factor
    
    def update_reputation(self, successful_repayment):
        """Update reputation score based on repayment success."""
        if successful_repayment:
            self.successful_repayments += 1
            # Reputation increases more slowly for already high-reputation agents
            increase_amount = 0.05 * (1.0 - self.reputation_score * 0.5)  # 0.025-0.05 increase
            self.reputation_score = min(1.0, self.reputation_score + increase_amount)
        else:
            self.defaults += 1
            # Reputation decreases more quickly for high-reputation agents (higher expectations)
            decrease_amount = 0.2 * (1.0 + self.reputation_score)  # 0.2-0.4 decrease
            self.reputation_score = max(0.0, self.reputation_score - decrease_amount)
        
        # Print debug info for significant reputation changes
        if successful_repayment and self.successful_repayments % 5 == 0:
            print(f"Agent {self.unique_id} reputation increased to {self.reputation_score:.2f}")
        elif not successful_repayment:
            print(f"Agent {self.unique_id} reputation decreased to {self.reputation_score:.2f}")
    
    def step(self):
        """
        At each step, the borrower:
        1. May take new loans if conditions are favorable
        2. Attempts to repay existing loans
        3. May get liquidated if collateral ratio falls below threshold
        """
        if self.is_liquidated:
            return
        
        self.time_since_last_borrow += 1
            
        # Check if we should be liquidated
        if self.borrowed_amount > 0:
            current_collateral_ratio = self.collateral / self.borrowed_amount
            # Get dynamic liquidation threshold based on reputation
            liquidation_threshold = self.model.get_liquidation_threshold(self)
            
            if current_collateral_ratio < liquidation_threshold:
                self.is_liquidated = True
                print(f"Agent {self.unique_id} LIQUIDATED! Collateral ratio: {current_collateral_ratio:.2f}, threshold: {liquidation_threshold:.2f}")
                self.model.liquidate_position(self)
                return
            
            # Randomly decrease collateral value (simulating price volatility)
            # Higher reputation agents experience less volatility (better risk management)
            volatility_chance = 0.05 * (1.0 - self.reputation_score * 0.5)  # 2.5-5% chance based on reputation
            if self.random.random() < volatility_chance:
                # Higher reputation agents experience smaller price drops
                min_decrease = 0.85 + (self.reputation_score * 0.1)  # 85-95% of original value
                decrease_factor = self.random.uniform(min_decrease, 0.98)
                self.collateral *= decrease_factor
                if self.collateral / self.borrowed_amount < liquidation_threshold * 1.1:  # Getting close to liquidation
                    print(f"Agent {self.unique_id} collateral dropped to {self.collateral:.2f}, ratio: {self.collateral/self.borrowed_amount:.2f}, threshold: {liquidation_threshold:.2f}")
        
        # Decide whether to borrow
        if self.borrowed_amount == 0 and self.time_since_last_borrow > 10:  # Reduced waiting time
            max_borrow = self.get_max_borrow_amount()
            available_liquidity = self.model.get_available_liquidity()
            
            # More aggressive borrowing based on reputation and sensitivity
            if self.model.reputation_sensitivity == 0:
                # Control case: more conservative borrowing
                target_ratio = 2.2  # Target a higher collateral ratio (more conservative)
                max_borrow = self.collateral / target_ratio
                borrow_factor = 0.8  # Use 80% of the max
                
                # Control case: less likely to borrow when utilization is high
                utilization_rate = self.model.get_utilization_rate()
                if utilization_rate > 0.5 and self.random.random() < 0.5:
                    return  # 50% chance to skip borrowing when utilization is high
            else:
                # With reputation system, higher reputation = more aggressive borrowing
                # Higher sensitivity means reputation has more impact
                borrow_factor = 0.6 + (self.reputation_score * 0.3 * (1 + self.model.reputation_sensitivity))
                borrow_factor = min(0.95, borrow_factor)  # Cap at 95%
                
                # Reputation-based systems: more likely to borrow even at high utilization
                # This creates higher utilization rates for reputation-based systems
                
            desired_borrow = max_borrow * borrow_factor
            borrow_amount = min(desired_borrow, available_liquidity)
            
            if borrow_amount > 50:  # Lower threshold to encourage borrowing
                success = self.model.process_borrow(borrow_amount)
                if success:
                    self.borrowed_amount = borrow_amount
                    self.time_since_last_borrow = 0
                    # Calculate initial collateral ratio
                    initial_ratio = self.collateral / borrow_amount
                    # Get liquidation threshold for this borrower
                    threshold = self.model.get_liquidation_threshold(self)
                    print(f"Agent {self.unique_id} borrowed {borrow_amount:.2f} with collateral {self.collateral:.2f}, ratio: {initial_ratio:.2f}, threshold: {threshold:.2f}")
        
        # Attempt repayment with some probability
        elif self.borrowed_amount > 0:
            # Higher reputation scores are more likely to repay
            if self.model.reputation_sensitivity == 0:
                # Control case: higher repayment probability but lower success rate
                repayment_probability = 0.05  # Fixed 5% chance
                repayment_success_rate = 0.5  # Fixed 50% success rate
            else:
                # With reputation system, dynamic repayment behavior
                repayment_probability = 0.02 + (self.reputation_score * 0.08)  # 2-10% chance based on reputation
                repayment_success_rate = 0.4 + self.reputation_score * 0.5  # 40-90% success rate
                
            if self.random.random() < repayment_probability:
                # Determine if repayment is successful
                repayment_success = self.random.random() < repayment_success_rate
                if repayment_success:
                    print(f"Agent {self.unique_id} repaid {self.borrowed_amount:.2f}")
                    self.model.process_repayment(self.borrowed_amount)
                    self.update_reputation(True)
                    self.borrowed_amount = 0
                else:
                    self.update_reputation(False) 