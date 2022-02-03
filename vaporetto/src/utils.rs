use std::cell::RefCell;
use std::collections::BTreeMap;

pub trait AddWeight {
    fn add_weight(&self, target: &mut [i32], offset: isize);
}

impl AddWeight for Vec<i32> {
    fn add_weight(&self, ys: &mut [i32], offset: isize) {
        if offset >= 0 {
            if let Some(ys) = ys.get_mut(offset as usize..) {
                for (w, y) in self.iter().zip(ys) {
                    *y += w;
                }
            }
        } else if let Some(ws) = self.get(-offset as usize..) {
            for (w, y) in ws.iter().zip(ys.iter_mut()) {
                *y += w;
            }
        }
    }
}

pub trait MergableWeight {
    fn from_two_weights(weight1: &Self, weight2: &Self, n_classes: usize) -> Self;
}

pub struct WeightMerger<W> {
    map: BTreeMap<String, RefCell<(W, bool)>>,
    n_classes: usize,
}

impl<W> WeightMerger<W>
where
    W: MergableWeight,
{
    pub fn new(n_classes: usize) -> Self {
        Self {
            map: BTreeMap::new(),
            n_classes,
        }
    }

    pub fn add(&mut self, ngram: &str, weight: W) {
        if let Some(data) = self.map.get_mut(ngram) {
            let (prev_weight, _) = &mut *data.borrow_mut();
            *prev_weight = W::from_two_weights(&weight, prev_weight, self.n_classes);
        } else {
            self.map
                .insert(ngram.to_string(), RefCell::new((weight, false)));
        }
    }

    pub fn merge(self) -> Vec<(String, W)> {
        let mut stack = vec![];
        for (ngram, data) in &self.map {
            if data.borrow().1 {
                continue;
            }
            stack.push(data);
            for (j, _) in ngram.char_indices().skip(1) {
                if let Some(data) = self.map.get(&ngram[j..]) {
                    stack.push(data);
                    if data.borrow().1 {
                        break;
                    }
                }
            }
            let mut data_from = stack.pop().unwrap();
            data_from.borrow_mut().1 = true;
            while let Some(data_to) = stack.pop() {
                let new_data = (
                    W::from_two_weights(&data_from.borrow().0, &data_to.borrow().0, self.n_classes),
                    true,
                );
                *data_to.borrow_mut() = new_data;
                data_from = data_to;
            }
        }
        self.map
            .into_iter()
            .map(|(ngram, weight)| (ngram, weight.into_inner().0))
            .collect()
    }
}

pub fn xor_or_zip_with<T, F>(lhs: &Option<T>, rhs: &Option<T>, f: F) -> Option<T>
where
    T: Clone,
    F: FnOnce(&T, &T) -> T,
{
    lhs.as_ref().map_or_else(
        || rhs.clone(),
        |x1| Some(rhs.as_ref().map_or_else(|| x1.clone(), |x2| f(x1, x2))),
    )
}
