//! Aho-Corasick algorithm using Double-Array Trie.

use std::collections::VecDeque;

struct SparseTrie {
    nodes: Vec<Vec<(u8, usize)>>,
    pattern_id: Vec<usize>,
    len: usize,
}

impl SparseTrie {
    fn new() -> Self {
        Self {
            nodes: vec![vec![]],
            pattern_id: vec![std::usize::MAX],
            len: 0,
        }
    }

    fn add(&mut self, pattern: &[u8]) -> usize {
        let mut node_id = 0;
        let prev_n_nodes = self.nodes.len();
        for &c in pattern {
            node_id = if let Some(next_node_id) = self.get(node_id, c) {
                next_node_id
            } else {
                let next_node_id = self.nodes.len();
                self.nodes.push(vec![]);
                self.nodes[node_id].push((c, next_node_id));
                self.pattern_id.push(std::usize::MAX);
                next_node_id
            };
        }
        if prev_n_nodes == self.nodes.len() {
            panic!("failed to add");
        }
        self.pattern_id[node_id] = self.len;
        self.len += 1;
        node_id
    }

    fn get(&self, node_id: usize, c: u8) -> Option<usize> {
        for trans in &self.nodes[node_id] {
            if c == trans.0 {
                return Some(trans.1);
            }
        }
        None
    }
}

pub struct Match {
    start: usize,
    end: usize,
    pattern: usize,
}

impl Match {
    pub const fn start(&self) -> usize {
        self.start
    }

    pub const fn end(&self) -> usize {
        self.end
    }

    pub const fn pattern(&self) -> usize {
        self.pattern
    }
}

pub struct DoubleArrayAhoCorasickIterator<'a, P>
where
    P: AsRef<[u8]>,
{
    pma: &'a DoubleArrayAhoCorasick,
    haystack: P,
    state_id: usize,
    pos: usize,
    cs_idx: usize,
    cs_pattern_ids: Option<&'a [usize]>
}

impl<'a, P> Iterator for DoubleArrayAhoCorasickIterator<'a, P>
where
    P: AsRef<[u8]>,
{
    type Item = Match;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(cs_pattern_ids) = self.cs_pattern_ids {
            if let Some(&pattern) = cs_pattern_ids.get(self.cs_idx) {
                self.cs_idx += 1;
                return Some(Match {
                    start: self.pos - self.pma.pattern_len[pattern],
                    end: self.pos,
                    pattern,
                });
            }
        }
        let haystack = self.haystack.as_ref();
        for (pos, &c) in haystack.iter().enumerate().skip(self.pos) {
            self.state_id = self.pma.get_next_state_id(self.state_id, c);
            if self.pma.pattern_ids[self.state_id] != std::usize::MAX {
                self.pos = pos + 1;
                let pattern = self.pma.pattern_ids[self.state_id];
                self.cs_idx = 0;
                self.cs_pattern_ids = self.pma.cs_pattern_ids.as_ref().map(|cs_pattern_ids| cs_pattern_ids[pattern].as_ref());
                return Some(Match {
                    start: self.pos - self.pma.pattern_len[pattern],
                    end: self.pos,
                    pattern,
                });
            }
        }
        self.pos = haystack.len();
        None
    }
}

pub struct DoubleArrayAhoCorasick {
    base: Vec<isize>,
    check: Vec<usize>,
    fail: Vec<usize>,
    pattern_ids: Vec<usize>,
    pattern_len: Vec<usize>,
    cs_pattern_ids: Option<Vec<Vec<usize>>>,
}

impl DoubleArrayAhoCorasick {
    pub fn new<D, P>(dict: D) -> Self
    where
        D: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        DoubleArrayAhoCorasickBuilder::new(65536, 65536).build(dict)
    }

    pub fn find_iter<P>(&self, haystack: P) -> DoubleArrayAhoCorasickIterator<P>
    where
        P: AsRef<[u8]>,
    {
        DoubleArrayAhoCorasickIterator {
            pma: self,
            haystack,
            state_id: 0,
            pos: 0,
            cs_idx: 0,
            cs_pattern_ids: None,
        }
    }

    fn get_child_index(&self, state_id: usize, c: u8) -> Option<usize> {
        let child_idx = self.base[state_id] + c as isize;
        if child_idx >= 0 {
            if let Some(&check) = self.check.get(child_idx as usize) {
                if check == state_id {
                    return Some(child_idx as usize);
                }
            }
        }
        None
    }

    fn get_next_state_id(&self, state_id: usize, c: u8) -> usize {
        let mut state_id = state_id;
        loop {
            if let Some(state_id) = self.get_child_index(state_id, c) {
                return state_id;
            } else {
                if state_id == 0 {
                    return 0;
                }
                state_id = self.fail[state_id];
            }
        }
    }
}

pub struct DoubleArrayAhoCorasickBuilder {
    base: Vec<isize>,
    check: Vec<usize>,
    fail: Vec<usize>,
    pattern_ids: Vec<usize>,
    pattern_len: Vec<usize>,
    cs_pattern_ids: Option<Vec<Vec<usize>>>,
    step_size: usize,
}

impl DoubleArrayAhoCorasickBuilder {
    pub fn new(init_size: usize, step_size: usize) -> Self {
        Self {
            base: vec![std::isize::MIN; init_size],
            check: vec![std::usize::MAX; init_size],
            pattern_ids: vec![std::usize::MAX; init_size],
            cs_pattern_ids: Some(vec![]),
            pattern_len: vec![],
            fail: vec![std::usize::MAX; init_size],
            step_size,
        }
    }

    pub fn match_shorter_suffix(mut self, flag: bool) -> Self {
        if flag {
            self.cs_pattern_ids.replace(vec![]);
        } else {
            self.cs_pattern_ids.take();
        };
        self
    }

    pub fn build<D, P>(mut self, dict: D) -> DoubleArrayAhoCorasick
    where
        D: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let sparse_trie = self.build_sparse_tree(dict);
        self.build_double_array(&sparse_trie);
        self.add_fails(&sparse_trie);

        let DoubleArrayAhoCorasickBuilder {
            base,
            check,
            fail,
            pattern_ids,
            pattern_len,
            cs_pattern_ids,
            ..
        } = self;
        DoubleArrayAhoCorasick {
            base,
            check,
            fail,
            pattern_ids,
            pattern_len,
            cs_pattern_ids,
        }
    }

    fn build_sparse_tree<D, P>(&mut self, dict: D) -> SparseTrie
    where
        D: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let mut trie = SparseTrie::new();
        for word in dict.into_iter() {
            let word = word.as_ref();
            trie.add(word);
            if let Some(cs_pattern_ids) = self.cs_pattern_ids.as_mut() {
                cs_pattern_ids.push(vec![]);
            };
            self.pattern_len.push(word.len());
        }
        trie
    }

    fn build_double_array(&mut self, sparse_trie: &SparseTrie) {
        let mut node_id_map = vec![std::usize::MAX; sparse_trie.nodes.len()];
        let mut min_idx = 1;
        let mut act_size = 1;
        node_id_map[0] = 0;
        self.check[0] = 0;
        for (i, node) in sparse_trie.nodes.iter().enumerate() {
            if node.is_empty() {
                continue;
            }
            let min_c = node[0].0;
            let mut base = min_idx - min_c as isize;
            'outer: loop {
                for &(c, _) in node {
                    let idx = (base + c as isize) as usize;
                    if idx + 1 > act_size {
                        act_size = idx + 1;
                    }
                    self.extend_arrays(idx + 1);
                    if self.check[idx] != std::usize::MAX {
                        if c == min_c {
                            min_idx += 1;
                        }
                        base += 1;
                        continue 'outer;
                    }
                }
                break;
            }
            for &(c, child_id) in node {
                let idx = (base + c as isize) as usize;
                self.check[idx] = node_id_map[i];
                self.pattern_ids[idx] = sparse_trie.pattern_id[child_id];
                node_id_map[child_id] = idx;
            }
            self.base[node_id_map[i]] = base;
        }
        self.truncate_arrays(act_size);
    }

    fn add_fails(&mut self, sparse_trie: &SparseTrie) {
        let mut queue = VecDeque::new();
        self.fail[0] = 0;
        for &(c, orig_child_idx) in &sparse_trie.nodes[0] {
            let child_idx = self.get_child_index(0, c).unwrap();
            self.fail[child_idx] = 0;
            queue.push_back((child_idx, orig_child_idx));
        }
        while let Some((node_idx, orig_node_idx)) = queue.pop_front() {
            for &(c, orig_child_idx) in &sparse_trie.nodes[orig_node_idx] {
                let child_idx = self.get_child_index(node_idx, c).unwrap();
                let mut fail_idx = self.fail[node_idx];
                loop {
                    if let Some(child_fail_idx) = self.get_child_index(fail_idx, c) {
                        self.fail[child_idx] = child_fail_idx;
                        if self.pattern_ids[child_fail_idx] != std::usize::MAX {
                            if self.pattern_ids[child_idx] == std::usize::MAX {
                                self.pattern_ids[child_idx] = self.pattern_ids[child_fail_idx];
                            } else if let Some(cs_pattern_ids) =
                                self.cs_pattern_ids.as_mut()
                            {
                                let child_pattern_id = self.pattern_ids[child_idx];
                                let fail_pattern_id = self.pattern_ids[child_fail_idx];
                                let mut fail_ids = cs_pattern_ids[fail_pattern_id].clone();
                                cs_pattern_ids[child_pattern_id].push(fail_pattern_id);
                                cs_pattern_ids[child_pattern_id].append(&mut fail_ids);
                            }
                        }
                        break;
                    } else {
                        let next_fail_idx = self.fail[fail_idx];
                        if fail_idx == 0 && next_fail_idx == 0 {
                            self.fail[child_idx] = 0;
                            break;
                        }
                        fail_idx = next_fail_idx;
                    }
                }
                queue.push_back((child_idx, orig_child_idx));
            }
        }
    }

    fn extend_arrays(&mut self, min_size: usize) {
        if min_size > self.base.len() {
            let new_len = ((min_size - self.base.len() - 1) / self.step_size + 1) * self.step_size
                + self.base.len();
            self.base.resize(new_len, std::isize::MIN);
            self.check.resize(new_len, std::usize::MAX);
            self.pattern_ids.resize(new_len, std::usize::MAX);
            self.fail.resize(new_len, std::usize::MAX);
        }
    }

    fn truncate_arrays(&mut self, size: usize) {
        self.base.truncate(size);
        self.check.truncate(size);
        self.pattern_ids.truncate(size);
        self.fail.truncate(size);
    }

    fn get_child_index(&self, idx: usize, c: u8) -> Option<usize> {
        let child_idx = self.base[idx] + c as isize;
        if child_idx >= 0 {
            if let Some(&check) = self.check.get(child_idx as usize) {
                if check == idx {
                    return Some(child_idx as usize);
                }
            }
        }
        None
    }
}
