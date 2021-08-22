//! Aho-Corasick algorithm using Double-Array Trie.

use std::collections::BTreeMap;
use std::collections::VecDeque;

pub struct Match {
    start: usize,
    end: usize,
    pattern: usize,
}

impl Match {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn pattern(&self) -> usize {
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
    common_suffix_idx: usize,
}

impl<'a, P> Iterator for DoubleArrayAhoCorasickIterator<'a, P>
where
    P: AsRef<[u8]>,
{
    type Item = Match;

    fn next(&mut self) -> Option<Self::Item> {
        if self.common_suffix_idx >= 1 {
            if let Some(cs_pattern_ids) = self.pma.common_suffix_pattern_ids.as_ref() {
                let pattern = self.pma.pattern_ids[self.state_id];
                if self.common_suffix_idx <= cs_pattern_ids[pattern].len() {
                    let pattern = cs_pattern_ids[pattern][self.common_suffix_idx - 1];
                    self.common_suffix_idx += 1;
                    return Some(Match {
                        start: self.pos - self.pma.pattern_len[pattern],
                        end: self.pos,
                        pattern,
                    });
                }
            }
        }
        self.common_suffix_idx = 0;
        let haystack = self.haystack.as_ref();
        for (pos, &c) in haystack.iter().enumerate().skip(self.pos) {
            self.state_id = self.pma.get_next_state_id(self.state_id, c);
            if self.pma.pattern_ids[self.state_id] != std::usize::MAX {
                self.pos = pos + 1;
                let pattern = self.pma.pattern_ids[self.state_id];
                self.common_suffix_idx = 1;
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
    common_suffix_pattern_ids: Option<Vec<Vec<usize>>>,
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
            common_suffix_idx: 0,
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
    common_suffix_pattern_ids: Option<Vec<Vec<usize>>>,
    step_size: usize,
}

impl DoubleArrayAhoCorasickBuilder {
    pub fn new(init_size: usize, step_size: usize) -> Self {
        Self {
            base: vec![std::isize::MIN; init_size],
            check: vec![std::usize::MAX; init_size],
            pattern_ids: vec![std::usize::MAX; init_size],
            common_suffix_pattern_ids: Some(vec![]),
            pattern_len: vec![],
            fail: vec![std::usize::MAX; init_size],
            step_size,
        }
    }

    pub fn match_shorter_suffix(mut self, flag: bool) -> Self {
        if flag {
            self.common_suffix_pattern_ids = Some(vec![]);
        } else {
            self.common_suffix_pattern_ids = None;
        }
        self
    }

    pub fn build<D, P>(mut self, dict: D) -> DoubleArrayAhoCorasick
    where
        D: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let (tree, tmp_pattern_ids) = self.build_tree(dict);
        self.construct_double_array(&tree, &tmp_pattern_ids);
        self.add_fails(&tree);

        let DoubleArrayAhoCorasickBuilder {
            base,
            check,
            fail,
            pattern_ids,
            pattern_len,
            common_suffix_pattern_ids,
            ..
        } = self;
        DoubleArrayAhoCorasick {
            base,
            check,
            fail,
            pattern_ids,
            pattern_len,
            common_suffix_pattern_ids,
        }
    }

    fn build_tree<D, P>(&mut self, dict: D) -> (Vec<BTreeMap<u8, usize>>, Vec<usize>)
    where
        D: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let mut tree = vec![BTreeMap::new()];
        let mut pattern_ids = vec![std::usize::MAX];
        for (i, word) in dict.into_iter().enumerate() {
            let mut node_id = 0;
            let word = word.as_ref();
            for c in word {
                node_id = if let Some(&node_id) = tree[node_id].get(c) {
                    node_id
                } else {
                    let new_node_id = tree.len();
                    tree[node_id].insert(*c, new_node_id);
                    tree.push(BTreeMap::new());
                    pattern_ids.push(std::usize::MAX);
                    new_node_id
                };
            }
            pattern_ids[node_id] = i;
            if let Some(cs_pattern_ids) = self.common_suffix_pattern_ids.as_mut() {
                cs_pattern_ids.push(vec![]);
            };
            self.pattern_len.push(word.len());
        }
        (tree, pattern_ids)
    }

    fn construct_double_array(&mut self, tree: &[BTreeMap<u8, usize>], tmp_pattern_ids: &[usize]) {
        let mut node_id_map = vec![std::usize::MAX; tree.len()];
        let mut min_idx = 1;
        let mut act_size = 1;
        node_id_map[0] = 0;
        self.check[0] = 0;
        for (i, node) in tree.iter().enumerate() {
            if node.is_empty() {
                continue;
            }
            let mut min_c = std::u8::MAX;
            for &c in node.keys() {
                if c < min_c {
                    min_c = c;
                }
            }
            let mut base = min_idx - min_c as isize;
            loop {
                let mut is_available = true;
                for (i, &c) in node.keys().enumerate() {
                    let idx = (base + c as isize) as usize;
                    if idx + 1 > act_size {
                        act_size = idx + 1;
                    }
                    self.extend_arrays(idx + 1);
                    if self.check[idx] != std::usize::MAX {
                        is_available = false;
                        if i == 0 {
                            min_idx += 1;
                        }
                        break;
                    }
                }
                if is_available {
                    for (&c, &child_id) in node {
                        self.check[(base + c as isize) as usize] = node_id_map[i];
                        self.pattern_ids[(base + c as isize) as usize] = tmp_pattern_ids[child_id];
                        node_id_map[child_id] = (base + c as isize) as usize;
                    }
                    break;
                }
                base += 1;
            }
            self.base[node_id_map[i]] = base;
        }
        self.truncate_arrays(act_size);
    }

    fn add_fails(&mut self, tree: &[BTreeMap<u8, usize>]) {
        let mut queue = VecDeque::new();
        self.fail[0] = 0;
        for (&c, &orig_child_idx) in &tree[0] {
            let child_idx = self.get_child_index(0, c).unwrap();
            self.fail[child_idx] = 0;
            queue.push_back((child_idx, orig_child_idx));
        }
        while let Some((node_idx, orig_node_idx)) = queue.pop_front() {
            for (&c, &orig_child_idx) in &tree[orig_node_idx] {
                let child_idx = self.get_child_index(node_idx, c).unwrap();
                let mut fail_idx = self.fail[node_idx];
                loop {
                    if let Some(child_fail_idx) = self.get_child_index(fail_idx, c) {
                        self.fail[child_idx] = child_fail_idx;
                        if self.pattern_ids[child_fail_idx] != std::usize::MAX {
                            if self.pattern_ids[child_idx] == std::usize::MAX {
                                self.pattern_ids[child_idx] = self.pattern_ids[child_fail_idx];
                            } else if let Some(cs_pattern_ids) =
                                self.common_suffix_pattern_ids.as_mut()
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
