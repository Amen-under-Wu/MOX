use std::fs::File;
use std::io::prelude::*;

fn read_input(path: &str) -> String {
    let mut file = File::open(path).expect("Unable to open file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Unable to read file");
    contents
}

fn tokenize(input: &str) -> Vec<Vec<String>> {
    input
        .lines()
        .map(|line| line.split_whitespace().map(String::from).collect())
        .filter(|tokens: &Vec<String>| !tokens.is_empty())
        .collect()
}

pub fn parse_input(path: &str) -> Vec<Vec<String>> {
    let input = read_input(path);
    tokenize(&input)
}
