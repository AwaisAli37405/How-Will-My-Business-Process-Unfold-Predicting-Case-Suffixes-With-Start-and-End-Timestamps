
import pandas as pd
import numpy as np
import utils.support as sup
import itertools
from operator import itemgetter
try:
    from support_modules import role_discovery as rl
except:
    import os
    from importlib import util
    spec = util.spec_from_file_location(
        'role_discovery', 
        os.path.join(os.getcwd(), 'support_modules', 'role_discovery.py'))
    rl = util.module_from_spec(spec)
    spec.loader.exec_module(rl)

class FeaturesMannager():


    def __init__(self, params):
        """constructor"""
        self.model_type = params['model_type']
        self.one_timestamp = params['one_timestamp']
        # self.resources = pd.DataFrame
        self.norm_method = params['norm_method']
        self._scalers = dict()
        self.scale_dispatcher = {'basic': self._scale_base,
                                 'inter': self._scale_inter}

    def calculate(self, log, add_cols):
        log = self.add_calculated_times(log)
        log = self.filter_features(log, add_cols)
        return self.scale_features(log, add_cols)

    @staticmethod
    def add_resources(log, rp_sim):
        # Resource pool discovery
        res_analyzer = rl.ResourcePoolAnalyser(log, sim_threshold=rp_sim)
        # Role discovery
        resources = pd.DataFrame.from_records(res_analyzer.resource_table)
        resources = resources.rename(index=str,
                                               columns={"resource": "user"})
        # Add roles information
        log = log.merge(resources, on='user', how='left')
        log = log[~log.task.isin(['Start', 'End'])]
        log = log.reset_index(drop=True)
        return log

    def convert_time_features_to_quarter(df: pd.DataFrame, time_columns: list) -> pd.DataFrame:
        """
        Converts a list of numeric time-based columns (in seconds since midnight)
        to quarter-of-hour indices (0–95).

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with time features in seconds.
        time_columns : list
            List of column names with time in seconds to convert.

        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with replaced columns containing quarter indices.
        """
        df = df.copy()

        for col in time_columns:
            # Convert seconds to quarter index (0–95)
            df[col] = df[col].apply(lambda s: int(s // 900) if pd.notnull(s) else np.nan)

        return df


    def filter_features(self, log, add_cols):
        # Add intercase features
        columns = ['caseid', 'task', 'user', 'end_timestamp', 
                   'role', 'dur', 'ac_index',  'rl_index']
        if not self.one_timestamp:
            columns.extend(['start_timestamp', 'wait'])
        columns.extend(add_cols)
        log = log[columns]
        return log

    @staticmethod
    def get_inter_arrival_time_case(log):

        # Ensure start_timestamp is in datetime format
        log['start_timestamp'] = pd.to_datetime(log['start_timestamp'])

        # Initialize the inter-arrival time column
        log['inter_arrival_time'] = 0

        # Extract unique cases and their start times
        start_times = log.groupby('caseid')['start_timestamp'].min().reset_index()
        start_times = start_times.sort_values(by='start_timestamp')

        # Calculate inter-arrival time for each case
        start_times['inter_arrival_time'] = start_times['start_timestamp'].diff().dt.total_seconds().fillna(0)

        # Map inter-arrival times to the original log
        inter_arrival_map = dict(zip(start_times['caseid'], start_times['inter_arrival_time']))
        log['inter_arrival_time'] = log['caseid'].map(inter_arrival_map)

        return log


    @staticmethod
    def get_active_cases(current_time, log):
        if isinstance(log, list):
            log = pd.DataFrame(log)
        # get the cases that are active at the current time
        active = log[(log['start_timestamp'] <= current_time) & (log['end_timestamp'] > current_time)]
        # get unique caseids
        active = active.drop_duplicates(subset='caseid')

        active_log = log[(log['start_timestamp'] <= current_time) & (log['end_timestamp'] > current_time)]
        return active, active_log

    @staticmethod
    def get_resource_availability(log, parms, resource_counts):
        if not parms['one_timestamp']:
            # Precompute resource counts for all events
            resource_counts_dict = resource_counts.set_index('ac_index')['unique_resource_count'].to_dict()

            events = log.to_dict('records')
            events = sorted(events, key=lambda x: x['caseid'])
            total_events = len(events)
            sup.print_progress(0, 'Analysing available resources')
            for i, event in enumerate(events):
                event_start_time = event['start_timestamp']
                active, active_log = FeaturesMannager.get_active_cases(event_start_time, log)
                event['active_cases_len'] = len(active)

                # Calculate locked resources
                locked_resources = active_log.groupby('ac_index').size().reset_index(name='locked_resources')
                locked_resources.columns = ['ac_index', 'locked_resources']
                locked_resources_dict = locked_resources.set_index('ac_index')['locked_resources'].to_dict()

                # Get total resources for the current activity
                ac_index = event['ac_index']
                total_resources = resource_counts_dict.get(ac_index, 0)
                event['total_resources'] = total_resources

                # Calculate available resources
                locked = locked_resources_dict.get(ac_index, 0)
                available_and_not_locked = max(total_resources - locked, 0)
                event['available_resources'] = available_and_not_locked

                # Update progress
                progress = ((i + 1) / total_events) * 100
                sup.print_progress(progress, 'Analysing available resources')
                # Convert events back to DataFrame
            log = pd.DataFrame(events)

        return log

    def read_multiple_json_files(file_paths: List[str]) -> List[Dict]:
        """

        Parameters:
        - file_paths (List[str]): A list of full paths to JSON files within Google Drive.

        Returns:
        - List[Dict]: A list of parsed JSON data as Python dictionaries.
        """


        data_list = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data_list.append(data)
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in {file_path}")
            except Exception as e:
                print(f"An unexpected error occurred while reading {file_path}: {e}")
        return data_list

    import json
    import pandas as pd

    def process_resource_calendar(file_path: str, output_csv: str = "resource_calendar.csv") -> pd.DataFrame:
        """
        Processes a JSON file containing resource calendar data and outputs a CSV file.

        Parameters:
        - file_path (str): Path to the input JSON file.
        - output_csv (str): Path to the output CSV file.

        Returns:
        - pd.DataFrame: Processed DataFrame containing resource calendar information.
        """
        # Load JSON data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

        # Extract resource calendar data
        resource_calendar_data = []
        for profile in data.get("resource_profiles", []):
            profile_name = profile.get("name")
            for resource in profile.get("resource_list", []):
                resource_name = resource.get("name")
                calendar_id = resource.get("calendar")
                calendar_details = next(
                    (calendar for calendar in data.get("resource_calendars", [])
                     if calendar.get("id") == calendar_id),
                    None
                )
                if calendar_details:
                    for period in calendar_details.get("time_periods", []):
                        day = period.get("from")
                        intervals = f"{period.get('beginTime')} - {period.get('endTime')}"
                        resource_calendar_data.append({
                            "Profile Name": profile_name,
                            "Resource": resource_name,
                            "Calendar ID": calendar_id,
                            "Day": day,
                            "Intervals": intervals
                        })

        # Create DataFrame
        calendar_df = pd.DataFrame(resource_calendar_data)

        # Split 'Intervals' into 'Start' and 'End'
        try:
            calendar_df[['Start', 'End']] = calendar_df['Intervals'].str.split(' - ', expand=True)
        except (AttributeError, KeyError):
            print("Error: 'Intervals' column is missing or not in the expected format.")
            return calendar_df

        # Convert 'Start' and 'End' to datetime.time objects, handling errors
        calendar_df['Start'] = pd.to_datetime(calendar_df['Start'], format='%H:%M:%S', errors='coerce').dt.time
        calendar_df['End'] = pd.to_datetime(calendar_df['End'], format='%H:%M:%S', errors='coerce').dt.time

        missing_count = calendar_df['End'].isnull().sum()
        if missing_count > 0:
            print(f"Replacing {missing_count} missing 'End' times with corresponding 'Start' times.")
            calendar_df['End'] = calendar_df['End'].fillna(calendar_df['Start'])

        # Save to CSV
        calendar_df.to_csv(output_csv, index=False)
        print(f"Resource calendar data saved to {output_csv}")

        return calendar_df

    # prompt: Extract unique users

    import pandas as pd

    def extract_unique_users(df):
        """
        Extracts unique users from a DataFrame, assuming a column named 'user_id' exists.
        """
        if 'user' not in df.columns:
            print("Error: 'user' column not found in the DataFrame.")
            return None  # or raise an exception

        unique_users = df['user'].unique()
        return unique_users

    # Assuming 'df' is your DataFrame (loaded from the CSV in the previous code)

    """### Implemented Methods to extract Features"""

    from datetime import datetime, timedelta
    import pandas as pd

    def time_till_next_onduty(df, calendar):
        """
        Calculates time until the next on-duty time for each activity instance,
        and returns debug info including user's calendar, current weekday, and time.
        """

        df = df.copy()
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], errors='coerce')
        df['user'] = df['user'].astype(str)

        cal = calendar.copy()
        cal['Resource'] = cal['Resource'].astype(str)
        cal['Day'] = cal['Day'].str.upper()

        day_map = {
            'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2,
            'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
        }
        num_to_day = {v: k for k, v in day_map.items()}

        cal['Day_Num'] = cal['Day'].map(day_map)
        cal['Start'] = pd.to_datetime(cal['Start'], format='%H:%M:%S', errors='coerce').dt.time
        cal['End'] = pd.to_datetime(cal['End'], format='%H:%M:%S', errors='coerce').dt.time

        def get_shift_info(row):
            user = row['user']
            ts = row['start_timestamp']
            if pd.isnull(ts):
                return pd.Series([-1, "NO_TIMESTAMP", "NO_DAY", "NO_TIME"])

            current_day = ts.weekday()
            current_time = ts.time()
            current_day_name = num_to_day[current_day]

            user_cal = cal[cal['Resource'] == user]
            calendar_info = user_cal[['Day', 'Start', 'End']].drop_duplicates().to_dict('records')

            if user_cal.empty:
                return pd.Series([-1, "NO_SHIFT_FOUND", current_day_name, current_time])

            # Step 1: Check if currently on duty
            today_shifts = user_cal[user_cal['Day_Num'] == current_day]
            for _, shift in today_shifts.iterrows():
                if pd.notnull(shift['Start']) and pd.notnull(shift['End']):
                    if shift['Start'] <= current_time < shift['End']:
                        return pd.Series([0, calendar_info, current_day_name, current_time])

            # Step 2: Check for any shift later today
            future_today_starts = []
            for _, shift in today_shifts.iterrows():
                if pd.notnull(shift['Start']) and shift['Start'] > current_time:
                    future_today_starts.append(shift['Start'])

            if future_today_starts:
                next_start_time = min(future_today_starts)
                next_shift_dt = datetime.combine(ts.date(), next_start_time)
                return pd.Series([(next_shift_dt - ts).total_seconds(), calendar_info, current_day_name, current_time])

            # Step 3: Check next days
            for offset in range(1, 8):
                check_day = (current_day + offset) % 7
                check_date = ts.date() + timedelta(days=offset)

                future_shifts = user_cal[user_cal['Day_Num'] == check_day]
                valid_starts = future_shifts['Start'].dropna()

                if not valid_starts.empty:
                    next_start_time = min(valid_starts)
                    next_shift_dt = datetime.combine(check_date, next_start_time)
                    return pd.Series(
                        [(next_shift_dt - ts).total_seconds(), calendar_info, current_day_name, current_time])

            # No future shifts at all
            return pd.Series([-1, calendar_info, current_day_name, current_time])

        result = df.apply(get_shift_info, axis=1)
        result.columns = [
            'seconds_until_next_onduty',
            'calendar_debug',
            'current_day_name',
            'current_time'
        ]

        return pd.concat([df, result], axis=1)



    def time_since_last_onduty(df, calendar):
        """
        For each row:
        - If the resource is off duty → return 0
        - If on duty → return seconds since start of that shift
        - If no shifts today → search previous days for the last shift and return time since it started

        Also returns current day/time and the resource's calendar for debugging.
        """

        df = df.copy()
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], errors='coerce')
        df['user'] = df['user'].astype(str)

        cal = calendar.copy()
        cal['Resource'] = cal['Resource'].astype(str)
        cal['Day'] = cal['Day'].str.upper()

        day_map = {
            'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2,
            'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
        }
        num_to_day = {v: k for k, v in day_map.items()}

        cal['Day_Num'] = cal['Day'].map(day_map)
        cal['Start'] = pd.to_datetime(cal['Start'], format='%H:%M:%S', errors='coerce').dt.time
        cal['End'] = pd.to_datetime(cal['End'], format='%H:%M:%S', errors='coerce').dt.time

        def get_shift_info(row):
            user = row['user']
            ts = row['start_timestamp']
            if pd.isnull(ts):
                return pd.Series(["NO_TIMESTAMP", "NO_SHIFT_FOUND", "NO_DAY", "NO_TIME"])

            current_day = ts.weekday()
            current_time = ts.time()
            current_day_name = num_to_day[current_day]

            user_cal = cal[cal['Resource'] == user]
            calendar_info = user_cal[['Day', 'Start', 'End']].drop_duplicates().to_dict('records')

            # Step 1: Check today's shifts
            today_shifts = user_cal[user_cal['Day_Num'] == current_day]
            for _, shift in today_shifts.iterrows():
                if pd.notnull(shift['Start']) and pd.notnull(shift['End']):
                    if shift['Start'] <= current_time < shift['End']:
                        shift_start_dt = datetime.combine(ts.date(), shift['Start'])
                        return pd.Series(
                            [(ts - shift_start_dt).total_seconds(), calendar_info, current_day_name, current_time])

            # Step 2: Not currently on duty → search back for most recent shift start
            for i in range(1, 8):  # Look back up to 7 days
                prev_day = (current_day - i) % 7
                prev_date = ts.date() - timedelta(days=i)
                prev_shifts = user_cal[user_cal['Day_Num'] == prev_day]

                shift_starts = []
                for _, shift in prev_shifts.iterrows():
                    if pd.notnull(shift['Start']):
                        shift_start_dt = datetime.combine(prev_date, shift['Start'])
                        shift_starts.append(shift_start_dt)

                if shift_starts:
                    last_onduty_time = max(shift_starts)
                    return pd.Series(
                        [(ts - last_onduty_time).total_seconds(), calendar_info, current_day_name, current_time])

            # No previous shifts found at all
            return pd.Series([-1, calendar_info, current_day_name, current_time])

        result = df.apply(get_shift_info, axis=1)
        result.columns = [
            'seconds_since_last_onduty',
            'calendar_debug',
            'current_day_name',
            'current_time'
        ]

        return pd.concat([df, result], axis=1)



    def time_until_offduty_debug(df, calendar):
        """
        For each row:
        - If resource is off duty → return 0
        - If on duty → return seconds until the shift ends

        Also returns current time, day, and resource calendar for debugging.
        """

        df = df.copy()
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], errors='coerce')
        df['user'] = df['user'].astype(str)

        cal = calendar.copy()
        cal['Resource'] = cal['Resource'].astype(str)
        cal['Day'] = cal['Day'].str.upper()

        day_map = {
            'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2,
            'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
        }
        num_to_day = {v: k for k, v in day_map.items()}

        cal['Day_Num'] = cal['Day'].map(day_map)
        cal['Start'] = pd.to_datetime(cal['Start'], format='%H:%M:%S', errors='coerce').dt.time
        cal['End'] = pd.to_datetime(cal['End'], format='%H:%M:%S', errors='coerce').dt.time

        def get_offduty_info(row):
            user = row['user']
            ts = row['start_timestamp']
            if pd.isnull(ts):
                return pd.Series([None, None, None, None])

            current_day = ts.weekday()
            current_time = ts.time()
            current_day_name = num_to_day[current_day]

            user_cal = cal[cal['Resource'] == user]
            calendar_info = user_cal[['Day', 'Start', 'End']].drop_duplicates().to_dict('records')

            today_shifts = user_cal[user_cal['Day_Num'] == current_day]
            for _, shift in today_shifts.iterrows():
                if pd.notnull(shift['Start']) and pd.notnull(shift['End']):
                    if shift['Start'] <= current_time < shift['End']:
                        # On duty: calculate time until off-duty
                        shift_end_dt = datetime.combine(ts.date(), shift['End'])
                        return pd.Series(
                            [(shift_end_dt - ts).total_seconds(), calendar_info, current_day_name, current_time])

            # Off duty
            return pd.Series([0, calendar_info, current_day_name, current_time])

        result = df.apply(get_offduty_info, axis=1)
        result.columns = [
            'seconds_until_offduty',
            'calendar_debug',
            'current_day_name',
            'current_time'
        ]

        return pd.concat([df, result], axis=1)


    def time_since_offduty_debug(df, calendar):
        """
        For each row:
        - If resource is on duty → return 0
        - If off duty → return seconds since the most recent off-duty time
        - If no shift today → go back to most recent shift on a previous day

        Returns debug columns to help validate the logic.
        """

        df = df.copy()
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], errors='coerce')
        df['user'] = df['user'].astype(str)

        cal = calendar.copy()
        cal['Resource'] = cal['Resource'].astype(str)
        cal['Day'] = cal['Day'].str.upper()

        day_map = {
            'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2,
            'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
        }
        num_to_day = {v: k for k, v in day_map.items()}

        cal['Day_Num'] = cal['Day'].map(day_map)
        cal['Start'] = pd.to_datetime(cal['Start'], format='%H:%M:%S', errors='coerce').dt.time
        cal['End'] = pd.to_datetime(cal['End'], format='%H:%M:%S', errors='coerce').dt.time

        def get_time_since_offduty(row):
            user = row['user']
            ts = row['start_timestamp']
            if pd.isnull(ts):
                return pd.Series([None, None, None, None])

            current_day = ts.weekday()
            current_time = ts.time()
            current_day_name = num_to_day[current_day]

            user_cal = cal[cal['Resource'] == user]
            calendar_info = user_cal[['Day', 'Start', 'End']].drop_duplicates().to_dict('records')

            # Today's shifts
            today_shifts = user_cal[user_cal['Day_Num'] == current_day]

            # Step 1: If currently on duty → return 0
            for _, shift in today_shifts.iterrows():
                if pd.notnull(shift['Start']) and pd.notnull(shift['End']):
                    if shift['Start'] <= current_time < shift['End']:
                        return pd.Series([0, calendar_info, current_day_name, current_time])

            # Step 2: Find the most recent shift that ended today (before current time)
            previous_ends = []
            for _, shift in today_shifts.iterrows():
                if shift['End'] < current_time:
                    shift_end_dt = datetime.combine(ts.date(), shift['End'])
                    previous_ends.append(shift_end_dt)

            if previous_ends:
                last_offduty_time = max(previous_ends)
                return pd.Series(
                    [(ts - last_offduty_time).total_seconds(), calendar_info, current_day_name, current_time])

            # Step 3: No shifts today or none before current time → search previous days
            for i in range(1, 8):  # Look back up to 7 days
                prev_day = (current_day - i) % 7
                prev_date = ts.date() - timedelta(days=i)
                prev_shifts = user_cal[user_cal['Day_Num'] == prev_day]

                shift_ends = []
                for _, shift in prev_shifts.iterrows():
                    if pd.notnull(shift['End']):
                        shift_end_dt = datetime.combine(prev_date, shift['End'])
                        shift_ends.append(shift_end_dt)

                if shift_ends:
                    last_offduty_time = max(shift_ends)
                    return pd.Series(
                        [(ts - last_offduty_time).total_seconds(), calendar_info, current_day_name, current_time])

            # No previous shifts found at all
            return pd.Series([-1, calendar_info, current_day_name, current_time])

        result = df.apply(get_time_since_offduty, axis=1)
        result.columns = [
            'seconds_since_last_offduty',
            'calendar_debug',
            'current_day_name',
            'current_time'
        ]

        return pd.concat([df, result], axis=1)

    def add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['dur'] = 0
        log['acc_cycle'] = 0
        log['daytime'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
            events = sorted(events, key=itemgetter(ordk))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace
                # is taken as instantsince there is no previous timestamp
                # to find a range
                if self.one_timestamp:
                    if i == 0:
                        dur = 0
                        acc = 0
                    else:
                        dur = (events[i]['end_timestamp'] -
                               events[i-1]['end_timestamp']).total_seconds()
                        acc = (events[i]['end_timestamp'] -
                               events[0]['end_timestamp']).total_seconds()
                else: #going to change this part dur and waiting time
                    dur = (events[i]['end_timestamp'] -
                           events[i]['start_timestamp']).total_seconds()
                    acc = (events[i]['end_timestamp'] -
                           events[0]['start_timestamp']).total_seconds()
                    if i == 0:
                        wit = 0
                    else:
                        wit = (events[i]['start_timestamp'] -
                               events[i-1]['start_timestamp']).total_seconds()
                    events[i]['wait'] = wit if wit >= 0 else 0
                events[i]['dur'] = dur
                events[i]['acc_cycle'] = acc
                time = events[i][ordk].time()
                time = time.second + time.minute*60 + time.hour*3600
                events[i]['daytime'] = time
                if self.one_timestamp:
                    events[i]['weekday'] = events[i]['end_timestamp'].weekday()
                else:
                    events[i]['weekday'] = events[i]['start_timestamp'].weekday()
        return pd.DataFrame.from_dict(log)

    def scale_features(self, log, add_cols):
        scaler = self._get_scaler(self.model_type)
        return scaler(log, add_cols)

    def register_scaler(self, model_type, scaler):
        try:
            self._scalers[model_type] = self.scale_dispatcher[scaler]
        except KeyError:
            raise ValueError(scaler)

    def _get_scaler(self, model_type):
        scaler = self._scalers.get(model_type)
        if not scaler:
            raise ValueError(model_type)
        return scaler

    def _scale_base(self, log, add_cols):
        if self.one_timestamp:
            log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        else:
            log, dur_scale = self.scale_feature(log, 'dur', self.norm_method)
            log, wait_scale = self.scale_feature(log, 'wait', self.norm_method)
            scale_args = {'dur': dur_scale, 'wait': wait_scale}
        return log, scale_args

    def _scale_inter(self, log, add_cols):
        # log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        if self.one_timestamp:
            log, scale_args = self.scale_feature(log, 'dur', self.norm_method)
        else:
            log, dur_scale = self.scale_feature(log, 'dur', self.norm_method)
            log, wait_scale = self.scale_feature(log, 'wait', self.norm_method)
            scale_args = {'dur': dur_scale, 'wait': wait_scale}
        for col in add_cols:
            if col == 'daytime':
                log, _ = self.scale_feature(log, 'daytime', 'day_secs', True)
            elif col == 'weekday':
                continue
            else:
                log, _ = self.scale_feature(log, col, self.norm_method, True)
        return log, scale_args

    # =========================================================================
    # Scale features
    # =========================================================================
    @staticmethod
    def scale_feature(log, feature, method, replace=False):
        """Scales a number given a technique.
        Args:
            log: Event-log to be scaled.
            feature: Feature to be scaled.
            method: Scaling method max, lognorm, normal, per activity.
            replace (optional): replace the original value or keep both.
        Returns:
            Scaleded value between 0 and 1.
        """
        scale_args = dict()
        if method == 'lognorm':
            log[feature + '_log'] = np.log1p(log[feature])
            max_value = np.max(log[feature+'_log'])
            min_value = np.min(log[feature+'_log'])
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature+'_log'], min_value), (max_value - min_value))
            log = log.drop((feature + '_log'), axis=1)
            scale_args = {'max_value': max_value, 'min_value': min_value}
        elif method == 'normal':
            max_value = np.max(log[feature])
            min_value = np.min(log[feature])
            log[feature+'_norm'] = np.divide(
                    np.subtract(log[feature], min_value), (max_value - min_value))
            scale_args = {'max_value': max_value, 'min_value': min_value}
        elif method == 'standard':
            mean = np.mean(log[feature])
            std = np.std(log[feature])
            log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
                                               std)
            scale_args = {'mean': mean, 'std': std}
        elif method == 'max':
            max_value = np.max(log[feature])
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
            scale_args = {'max_value': max_value}
        elif method == 'day_secs':
            max_value = 86400
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
            scale_args = {'max_value': max_value}
        elif method is None:
            log[feature+'_norm'] = log[feature]
        else:
            raise ValueError(method)
        if replace:
            log = log.drop(feature, axis=1)
        return log, scale_args
