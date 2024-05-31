import argparse
import pandas as pd
from pulp import *

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def read_data(input_path, output_path):
    df_input = pd.read_csv(input_path, sep=';')
    input_columns = list(df_input.columns[1:])
    df_input.columns = ['name'] + input_columns

    df_output = pd.read_csv(output_path, sep=';')
    output_columns = list(df_output.columns[1:])
    df_output.columns = ['name'] + output_columns

    df = pd.merge(df_input, df_output, on='name')
    return df, input_columns, output_columns


def efficiency(df, input_columns, output_columns):
    solutions = {}
    for idx_dmu, dmu in df.iterrows():
        
        problem = LpProblem("dea", LpMinimize)
        
        theta = LpVariable("theta", 0)
        
        decision_variables = {idx: LpVariable(f"x_{idx}", 0) for idx in df.index}

        for column in df.columns:
            if column in input_columns:
                problem += lpSum(value * decision_variables[idx] for value, idx in zip(df[column], df.index)) <= dmu[column] * theta
            if column in output_columns:
                problem += lpSum(value * decision_variables[idx] for value, idx in zip(df[column], df.index)) >= dmu[column]
    
        problem += theta
        problem.solve(solver=GLPK(msg=False))
        solution = {variable.name: variable.varValue for variable in problem.variables()}
        solutions[idx_dmu] = solution

    results = {idx: solution for idx, solution in solutions.items()}
    for idx, sol in results.items():
        print(f"{df.loc[idx, 'name']} & {sol['theta']:.3f} \\\\")
    return results


def hcu(df, input_columns, output_columns, results):
    for idx, sol in results.items():
        if sol['theta'] < 1:  
            print(df.loc[idx, 'name'], end="")
            for input_column in input_columns:
                hcu_value = sum(value * _lambda for value, _lambda in zip(df[input_column], [v for k, v in sorted([(k, v) for k, v in sol.items() if k.startswith('x')], key=lambda x: int(x[0].split('_')[1]))]))
                print(f" & {hcu_value:.3f}", end="")
            for input_column in input_columns:
                hcu_value = sum(value * _lambda for value, _lambda in zip(df[input_column], [v for k, v in sorted([(k, v) for k, v in sol.items() if k.startswith('x')], key=lambda x: int(x[0].split('_')[1]))]))
                improvement = df.loc[idx, input_column] - hcu_value
                print(f" & {improvement:.3f}", end="")    
            print(" \\\\")


def super_efficiency(df, input_columns, output_columns):
    solutions = {}
    for idx_dmu in df.index:
        
        problem = LpProblem("dea", LpMaximize)
        
        decision_variables_v = {column: LpVariable(f"v_{column}", 0) for column in input_columns}
        decision_variables_u = {column: LpVariable(f"u_{column}", 0) for column in output_columns}
        
        problem += lpSum(value * variable for value, variable in zip(df.loc[idx_dmu, input_columns], decision_variables_v.values())) == 1

        for idx, dmu2 in df.iterrows():
            if idx == idx_dmu:
                continue
            problem += lpSum(value * variable for value, variable in zip(df.loc[idx, output_columns], decision_variables_u.values())) <= lpSum(value * variable for value, variable in zip(df.loc[idx, input_columns], decision_variables_v.values()))

        problem += lpSum(value * variable for value, variable in zip(df.loc[idx_dmu, output_columns], decision_variables_u.values()))
        problem.solve(solver=GLPK(msg=False))
        solution = {variable.name: variable.varValue for variable in problem.variables()}
        solutions[idx_dmu] = solution

    df_super = pd.DataFrame(0.0, index=df.index, columns=['super_eff'])
    for idx, solution in solutions.items():
        super_efficiency_value = sum(value * variable for value, variable in zip(df.loc[idx, output_columns], [v for k, v in solution.items() if k.startswith('u')]))
        df_super.loc[idx, 'super_eff'] = super_efficiency_value
        print(f"{df.loc[idx, 'name']} & {super_efficiency_value:.3f} \\\\")
    return df_super


def cross_efficiency(df, input_columns, output_columns):
    df_cross = pd.DataFrame(0.0, index=df.index, columns=df.index)
    sol = efficiency(df, input_columns, output_columns)
    for idx_dmu in df.index:
        decision_variables_v = {column: LpVariable(f"v_{column}", 0) for column in input_columns}
        decision_variables_u = {column: LpVariable(f"u_{column}", 0) for column in output_columns}

        sums_inputs = {column: df.drop(index=idx_dmu)[column].sum() for column in input_columns}
        sums_outputs = {column: df.drop(index=idx_dmu)[column].sum() for column in output_columns}

        problem = LpProblem("dea", LpMinimize)

        problem += lpSum(value * variable for value, variable in zip(sums_inputs.values(), decision_variables_v.values())) == 1

        for idx, dmu2 in df.iterrows():
            if idx == idx_dmu:
                problem += lpSum(value * variable for value, variable in zip(df.loc[idx, output_columns], decision_variables_u.values())) == sol[idx_dmu]['theta'] * lpSum(value * variable for value, variable in zip(df.loc[idx, input_columns], decision_variables_v.values()))
            else:
                problem += lpSum(value * variable for value, variable in zip(df.loc[idx, output_columns], decision_variables_u.values())) <= lpSum(value * variable for value, variable in zip(df.loc[idx, input_columns], decision_variables_v.values()))

        problem += lpSum(value * variable for value, variable in zip(sums_outputs.values(), decision_variables_u.values()))
        problem.solve(solver=GLPK(msg=False))
        solution = {variable.name: variable.varValue for variable in problem.variables()}

        for idx_dmu2 in df.index:
            if idx_dmu == idx_dmu2:
                df_cross.loc[idx_dmu, idx_dmu2] = sol[idx_dmu]['theta']
            else:
                numerator = sum(value * variable for value, variable in zip(df.loc[idx_dmu2, output_columns], [v for k, v in solution.items() if k.startswith('u')]))
                denominator = sum(value * variable for value, variable in zip(df.loc[idx_dmu2, input_columns], [v for k, v in solution.items() if k.startswith('v')]))
                df_cross.loc[idx_dmu, idx_dmu2] = numerator / denominator

    for idx in df_cross.index:
        print(df.loc[idx, 'name'], end="")
        for idx2 in df_cross.index:
            print(f" & {df_cross.loc[idx, idx2]:.3f}", end="")
        print(f" & {df_cross.mean()[idx]:.3f}", end="")
        print(" \\\\")
    return df_cross


def samples_efficiency(df, input_columns, output_columns, samples_path):
    df_samples = pd.read_csv(samples_path, sep=';')
    df_samples_eff = pd.DataFrame(0.0, index=df.index, columns=df_samples.index)

    for idx_sample in df_samples.index:
        for idx_dmu in df.index:
            numerator = sum(value * variable for value, variable in zip(df.loc[idx_dmu, output_columns], df_samples.loc[idx_sample, [col for col in df_samples.columns if col.startswith('o')]]))
            denominator = sum(value * variable for value, variable in zip(df.loc[idx_dmu, input_columns], df_samples.loc[idx_sample, [col for col in df_samples.columns if col.startswith('i')]]))
            df_samples_eff.loc[idx_dmu, idx_sample] = numerator / denominator

    df_samples_eff = df_samples_eff / df_samples_eff.max()
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    def count_bins(row):
        binned = pd.cut(row, bins=bins, labels=labels, include_lowest=True)
        return binned.value_counts().reindex(labels, fill_value=0)

    df_buckets = df_samples_eff.apply(count_bins, axis=1)
    df_buckets = df_buckets.div(df_buckets.sum(axis=1), axis=0)

    for idx in df_buckets.index:
        print(df.loc[idx, 'name'], end="")
        for col in df_buckets.columns:
            print(f" & {df_buckets.loc[idx, col]:.3f}", end="")
        print(f" & {df_samples_eff.mean(axis=1)[idx]:.3f}", end="")
        print(" \\\\")
    return df_samples_eff


def rankings(df, df_super, df_cross, df_samples_eff):
    for idx in df_super.sort_values('super_eff', ascending=False).index:
        print(f" {df.loc[idx, 'name']} \succ", end="")
    print("")
    for idx in df_cross.mean().sort_values(ascending=False).index:
        print(f" {df.loc[idx, 'name']} \succ", end="")
    print("")
    for idx in df_samples_eff.mean(axis=1).sort_values(ascending=False).index:
        print(f" {df.loc[idx, 'name']} \succ", end="")
    print("")


def main():
    parser = argparse.ArgumentParser(description='Data Envelopment Analysis')
    parser.add_argument('--part', choices=['efficiency', 'hcu', 'super-efficiency', 'cross-efficiency', 'samples-efficiency', 'rankings'], required=True, help='Part of the process to run')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')
    parser.add_argument('--samples', help='Path to the CSV samples file (required if part is samples-efficiency)')
    
    args = parser.parse_args()
    
    df, input_columns, output_columns = read_data(args.input, args.output)
    
    if args.part == 'efficiency':
        _ = efficiency(df, input_columns, output_columns)
    elif args.part == 'hcu':
        results = efficiency(df, input_columns, output_columns)
        hcu(df, input_columns, output_columns, results)
    elif args.part == 'super-efficiency':
        _ = super_efficiency(df, input_columns, output_columns)
    elif args.part == 'cross-efficiency':
        _ = cross_efficiency(df, input_columns, output_columns)
    elif args.part == 'samples-efficiency':
        if not args.samples:
            print("Samples file path is required if part is samples-efficiency")
            return
        df_samples_eff = samples_efficiency(df, input_columns, output_columns, args.samples)
    elif args.part == 'rankings':
        df_super = super_efficiency(df, input_columns, output_columns)
        df_cross = cross_efficiency(df, input_columns, output_columns)
        if args.samples:
            df_samples_eff = samples_efficiency(df, input_columns, output_columns, args.samples)
        else:
            print("Samples file path is required if part is rankings")
            return
        rankings(df, df_super, df_cross, df_samples_eff)

if __name__ == "__main__":
    main()
